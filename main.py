"""Interactive qualitative 2D E/B field sandbox."""
import json
import math
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
from PySide6.QtCore import Qt, QPointF, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush, QTransform
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QCheckBox, QFormLayout,
    QGroupBox, QMessageBox, QSpinBox, QDoubleSpinBox)

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

@dataclass
class Shape:
    kind: str
    x: float
    y: float
    rx: float = 75.0
    ry: float = 24.0
    angle: float = 0.0
    charge: int = 0
    power: float = 1.0


def point_in(o, x, y):
    a = math.radians(o.angle)
    dx, dy = x-o.x, y-o.y
    u = dx*math.cos(a)+dy*math.sin(a)
    v = -dx*math.sin(a)+dy*math.cos(a)
    return (u/o.rx)**2+(v/o.ry)**2 <= 1


def boundary(o, count=72):
    t = np.linspace(0, 2*np.pi, count, endpoint=False)
    a = math.radians(o.angle)
    c, s = np.cos(a), np.sin(a)
    u, v = o.rx*np.cos(t), o.ry*np.sin(t)
    x = o.x + c*u-s*v
    y = o.y + s*u+c*v
    return x, y, t


def touching(a, b):
    if point_in(a,b.x,b.y) or point_in(b,a.x,a.y):
        return True
    for o, other in ((a,b),(b,a)):
        xs, ys, _ = boundary(o, 64)
        if any(point_in(other, float(x), float(y)) for x,y in zip(xs,ys)):
            return True
    return False


def get_sources(shapes):
    metals=[o for o in shapes if o.kind=='Metal']
    parent=list(range(len(metals)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(len(metals)):
        for j in range(i):
            if touching(metals[i],metals[j]):
                parent[find(i)]=find(j)
    groups={}
    for i,o in enumerate(metals):
        groups.setdefault(find(i),[]).append(o)
    src=[]
    for group in groups.values():
        net=sum(o.charge for o in group)
        if not net: continue
        pts=[]
        for o in group:
            xs,ys,t=boundary(o)
            for x,y,theta in zip(xs,ys,t):
                if any(point_in(other,float(x),float(y)) for other in group if other is not o):
                    continue
                speed=np.hypot(o.rx*np.sin(theta),o.ry*np.cos(theta))
                curvature=(o.rx*o.ry)/max(speed**3,1e-6)
                weight=speed*(1+min(3,15*curvature))
                pts.append((x,y,weight))
        total=sum(p[2] for p in pts)
        if total:
            src.extend((x,y,net*w/total) for x,y,w in pts)
    charges=np.array(src,dtype=np.float64).reshape(-1,3)
    coils=np.array([[o.x,o.y,math.radians(o.angle),o.power,1 if o.kind=='Coil XY' else 2,max(o.rx,o.ry)] for o in shapes if o.kind.startswith('Coil')],dtype=np.float64).reshape(-1,6)
    return charges,coils


if HAS_NUMBA:
    @njit(parallel=True,cache=True)
    def field_kernel(xs,ys,charges,coils):
        n=len(xs)
        E=np.zeros(n)
        B=np.zeros(n)
        for i in prange(n):
            x,y=xs[i],ys[i]
            ex=ey=bx=by=bz=0.0
            for j in range(len(charges)):
                dx=x-charges[j,0];dy=y-charges[j,1]
                r2=dx*dx+dy*dy+100.0
                k=charges[j,2]*24000.0/(r2*math.sqrt(r2))
                ex+=dx*k;ey+=dy*k
            for j in range(len(coils)):
                dx=x-coils[j,0];dy=y-coils[j,1]
                a=coils[j,2];p=coils[j,3];radius=coils[j,5]
                r2=dx*dx+dy*dy+(radius*.65)**2
                if coils[j,4]==1:
                    c=math.cos(a);s=math.sin(a);dot=dx*c+dy*s
                    v=p*650000.0/(r2**1.5)
                    bx+=v*(3*dot*dx/r2-c)
                    by+=v*(3*dot*dy/r2-s)
                else:
                    bz+=p*650000.0*((radius*.7)**2-2*(dx*dx+dy*dy))/(r2**2.5)
            E[i]=math.sqrt(ex*ex+ey*ey)
            B[i]=math.sqrt(bx*bx+by*by+bz*bz)
        return E,B
else:
    def field_kernel(xs,ys,charges,coils):
        ex=np.zeros_like(xs);ey=np.zeros_like(xs)
        bx=np.zeros_like(xs);by=np.zeros_like(xs);bz=np.zeros_like(xs)
        for px,py,q in charges:
            dx=xs-px;dy=ys-py;r2=dx*dx+dy*dy+100
            k=q*24000/(r2*np.sqrt(r2));ex+=dx*k;ey+=dy*k
        for cx,cy,a,p,typ,radius in coils:
            dx=xs-cx;dy=ys-cy;r2=dx*dx+dy*dy+(radius*.65)**2
            if typ==1:
                c,s=np.cos(a),np.sin(a);dot=dx*c+dy*s;v=p*650000/(r2**1.5)
                bx+=v*(3*dot*dx/r2-c);by+=v*(3*dot*dy/r2-s)
            else:
                bz+=p*650000*((radius*.7)**2-2*(dx*dx+dy*dy))/(r2**2.5)
        return np.hypot(ex,ey),np.sqrt(bx*bx+by*by+bz*bz)


class Canvas(QWidget):
    selectionChanged=Signal()
    def __init__(self):
        super().__init__()
        self.setMinimumSize(720,480)
        self.setMouseTracking(True)
        self.shapes=[];self.selected=-1;self.mode='Select'
        self.zoom=1.;self.panx=0.;self.pany=0.
        self.show_e=True;self.show_b=True;self.gain_e=1.;self.gain_b=1.
        self.image=None;self.drag=None
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.compute)
        self.reset()
    def reset(self):
        self.shapes=[Shape('Metal',280,185,125,22,0,1),Shape('Metal',280,300,125,22,0,-1),Shape('Coil XY',530,240,50,25,90,0,1)]
        self.selected=-1;self.zoom=1.;self.panx=self.pany=0.;self.invalidate();self.selectionChanged.emit()
    def invalidate(self):
        self.timer.start(70);self.update()
    def world(self,p):
        return ((p.x()-self.panx)/self.zoom,(p.y()-self.pany)/self.zoom)
    def compute(self):
        w=max(120,int(self.width()/3));h=max(90,int(self.height()/3))
        w=min(w,450);h=min(h,300)
        xx=(np.arange(w)+.5)*self.width()/w;yy=(np.arange(h)+.5)*self.height()/h
        X,Y=np.meshgrid(xx,yy)
        xs=((X.ravel()-self.panx)/self.zoom).astype(np.float64)
        ys=((Y.ravel()-self.pany)/self.zoom).astype(np.float64)
        charges,coils=get_sources(self.shapes)
        E,B=field_kernel(xs,ys,charges,coils)
        # Independent display scaling: no physical comparison of E and B units.
        e=255*(1-np.exp(-np.power(np.maximum(E*self.gain_e/3,0),.65))) if self.show_e else np.zeros_like(E)
        b=255*(1-np.exp(-np.power(np.maximum(B*self.gain_b*10/3,0),.65))) if self.show_b else np.zeros_like(B)
        rgba=np.zeros((h,w,4),dtype=np.uint8)
        rgba[:,:,0]=np.clip(b.reshape(h,w),0,255).astype(np.uint8)
        rgba[:,:,1]=np.clip(e.reshape(h,w),0,255).astype(np.uint8)
        rgba[:,:,3]=255
        self.image=QImage(rgba.tobytes(),w,h,w*4,QImage.Format.Format_RGBA8888).copy()
        self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('black'))
        if self.image:p.drawImage(self.rect(),self.image)
        p.translate(self.panx,self.pany);p.scale(self.zoom,self.zoom)
        for i,o in enumerate(self.shapes):
            p.save();p.translate(o.x,o.y);p.rotate(o.angle)
            if o.kind=='Metal':fill='#9ca3af';outline='#e5e7eb'
            elif o.kind=='Insulator':fill='#f8fafc';outline='#ffffff'
            else:fill='#92400e';outline='#fdba74'
            p.setPen(QPen(QColor(outline),2/self.zoom));p.setBrush(QBrush(QColor(fill)))
            p.drawEllipse(QPointF(0,0),o.rx,o.ry)
            if o.kind=='Metal':
                p.setPen(QColor('#111827'));p.drawText(-12,-12,24,24,Qt.AlignmentFlag.AlignCenter,'+' if o.charge>0 else '−' if o.charge<0 else '0')
            elif o.kind.startswith('Coil'):
                p.setPen(QColor('#ffffff'));p.drawText(-20,-12,40,24,Qt.AlignmentFlag.AlignCenter,'⊙' if o.kind=='Coil Z' else '↔')
            if i==self.selected:
                p.setPen(QPen(QColor('#facc15'),2/self.zoom,Qt.PenStyle.DashLine));p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(0,0),o.rx+6,o.ry+6)
            p.restore()
        p.end()
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.MiddleButton:
            self.drag=('pan',event.position().x(),event.position().y(),self.panx,self.pany);return
        if event.button()!=Qt.MouseButton.LeftButton:return
        x,y=self.world(event.position());hit=-1
        for i in reversed(range(len(self.shapes))):
            if point_in(self.shapes[i],x,y):hit=i;break
        if self.mode!='Select' and hit==-1:
            kind=self.mode;self.shapes.append(Shape(kind,x,y,50 if kind.startswith('Coil') else 85,25))
            hit=len(self.shapes)-1;self.mode='Select'
        self.selected=hit
        self.drag=('move',x-self.shapes[hit].x,y-self.shapes[hit].y) if hit>=0 else None
        self.selectionChanged.emit();self.invalidate()
    def mouseMoveEvent(self,event):
        if not self.drag:return
        if self.drag[0]=='pan':
            _,sx,sy,px,py=self.drag
            self.panx=px+event.position().x()-sx;self.pany=py+event.position().y()-sy
        elif self.selected>=0:
            x,y=self.world(event.position());self.shapes[self.selected].x=x-self.drag[1];self.shapes[self.selected].y=y-self.drag[2]
        self.invalidate()
    def mouseReleaseEvent(self,event):self.drag=None
    def wheelEvent(self,event):
        old=self.world(event.position());self.zoom=max(.2,min(5,self.zoom*(1.15 if event.angleDelta().y()>0 else 1/1.15)))
        self.panx=event.position().x()-old[0]*self.zoom;self.pany=event.position().y()-old[1]*self.zoom;self.invalidate()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('E/B Field Sandbox — qualitative prototype');self.resize(1220,760)
        main=QWidget();self.setCentralWidget(main);row=QHBoxLayout(main)
        self.canvas=Canvas();row.addWidget(self.canvas,1)
        side=QWidget();side.setFixedWidth(285);layout=QVBoxLayout(side);row.addWidget(side)
        layout.addWidget(QLabel('ADD / SELECT OBJECT'))
        self.mode=QComboBox();self.mode.addItems(['Select','Metal','Insulator','Coil XY','Coil Z']);self.mode.currentTextChanged.connect(lambda s:setattr(self.canvas,'mode',s));layout.addWidget(self.mode)
        layout.addWidget(QLabel('Choose an item, then click blank canvas. Drag to move.'))
        self.group=QGroupBox('Selected object');form=QFormLayout(self.group);layout.addWidget(self.group)
        self.kind_label=QLabel('None');form.addRow('Type',self.kind_label)
        self.rx=QDoubleSpinBox();self.ry=QDoubleSpinBox()
        for widget in (self.rx,self.ry):widget.setRange(5,500);widget.setDecimals(0)
        self.angle=QDoubleSpinBox();self.angle.setRange(-180,180);self.angle.setSuffix('°')
        self.charge=QComboBox();self.charge.addItems(['Negative (−)','Neutral (0)','Positive (+)'])
        self.power=QDoubleSpinBox();self.power.setRange(-10,10);self.power.setSingleStep(.25)
        for label,widget in [('Width radius',self.rx),('Height radius',self.ry),('Rotation',self.angle),('Charge',self.charge),('Coil polarity / strength',self.power)]:form.addRow(label,widget)
        for widget,attr in ((self.rx,'rx'),(self.ry,'ry'),(self.angle,'angle'),(self.power,'power')):widget.valueChanged.connect(lambda value,a=attr:self.change(a,value))
        self.charge.currentIndexChanged.connect(lambda i:self.change('charge',i-1))
        delete=QPushButton('Delete selected');delete.clicked.connect(self.delete);form.addRow(delete)
        layout.addWidget(QLabel('FIELD DISPLAY'))
        self.showe=QCheckBox('Green: electric field |E|');self.showe.setChecked(True)
        self.showb=QCheckBox('Red: magnetic field |B|');self.showb.setChecked(True)
        for widget,attr in ((self.showe,'show_e'),(self.showb,'show_b')):
            widget.toggled.connect(lambda v,a=attr:(setattr(self.canvas,a,v),self.canvas.invalidate()))
            layout.addWidget(widget)
        for label,attr in [('Electric gain','gain_e'),('Magnetic gain','gain_b')]:
            layout.addWidget(QLabel(label));slider=QSlider(Qt.Orientation.Horizontal);slider.setRange(1,80);slider.setValue(10)
            slider.valueChanged.connect(lambda v,a=attr:(setattr(self.canvas,a,v/10),self.canvas.invalidate()));layout.addWidget(slider)
        save=QPushButton('Save scene JSON');save.clicked.connect(self.save);layout.addWidget(save)
        load=QPushButton('Load scene JSON');load.clicked.connect(self.load);layout.addWidget(load)
        clear=QPushButton('Clear scene');clear.clicked.connect(self.clear);layout.addWidget(clear)
        reset=QPushButton('Restore example');reset.clicked.connect(self.canvas.reset);layout.addWidget(reset)
        note=QLabel('CPU: Numba multicore JIT' if HAS_NUMBA else 'CPU: NumPy fallback')
        layout.addWidget(note)
        disclaimer=QLabel('Approximate static fields only. Not an accurate conductor, dielectric, solenoid, force, or torque solver.');disclaimer.setWordWrap(True);layout.addWidget(disclaimer)
        layout.addStretch()
        self.canvas.selectionChanged.connect(self.sync);self.sync()
        QTimer.singleShot(250,self.canvas.invalidate)
    def sync(self):
        i=self.canvas.selected;o=self.canvas.shapes[i] if 0<=i<len(self.canvas.shapes) else None
        self.kind_label.setText(o.kind if o else 'None')
        for widget,attr in ((self.rx,'rx'),(self.ry,'ry'),(self.angle,'angle'),(self.power,'power')):
            widget.blockSignals(True);widget.setEnabled(bool(o));widget.setValue(getattr(o,attr) if o else 0);widget.blockSignals(False)
        self.charge.blockSignals(True);self.charge.setEnabled(bool(o and o.kind=='Metal'));self.charge.setCurrentIndex(o.charge+1 if o else 1);self.charge.blockSignals(False)
        self.power.setEnabled(bool(o and o.kind.startswith('Coil')))
        if self.canvas.mode=='Select' and self.mode.currentText()!='Select':self.mode.setCurrentText('Select')
    def change(self,attr,value):
        i=self.canvas.selected
        if 0<=i<len(self.canvas.shapes):setattr(self.canvas.shapes[i],attr,value);self.canvas.invalidate()
    def delete(self):
        i=self.canvas.selected
        if 0<=i<len(self.canvas.shapes):self.canvas.shapes.pop(i);self.canvas.selected=-1;self.sync();self.canvas.invalidate()
    def clear(self):self.canvas.shapes=[];self.canvas.selected=-1;self.sync();self.canvas.invalidate()
    def save(self):
        path,_=QFileDialog.getSaveFileName(self,'Save scene','','JSON (*.json)')
        if path:
            with open(path,'w',encoding='utf-8') as f:json.dump([asdict(o) for o in self.canvas.shapes],f,indent=2)
    def load(self):
        path,_=QFileDialog.getOpenFileName(self,'Load scene','','JSON (*.json)')
        if not path:return
        try:
            with open(path,encoding='utf-8') as f:items=json.load(f)
            if not isinstance(items,list) or len(items)>300:raise ValueError('Expected list of at most 300 objects')
            allowed={'Metal','Insulator','Coil XY','Coil Z'}
            shapes=[]
            for item in items:
                o=Shape(**item)
                if o.kind not in allowed or not (5<=o.rx<=500 and 5<=o.ry<=500) or not all(math.isfinite(v) for v in (o.x,o.y,o.angle,o.power)):raise ValueError('Invalid object parameters')
                shapes.append(o)
            self.canvas.shapes=shapes;self.canvas.selected=-1;self.sync();self.canvas.invalidate()
        except Exception as exc:QMessageBox.warning(self,'Cannot load scene',str(exc))

if __name__=='__main__':
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    window=MainWindow();window.show()
    sys.exit(app.exec())
