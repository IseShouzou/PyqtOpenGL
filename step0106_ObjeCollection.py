import sys
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import *


#
#  Functions 
#
def calcLookAt( e, t, u ) :
    ''' Calculate LookAt Matrix

        Args:
            e : eye position
            t : target position
            u : upper direction

        Returns:
            LookAt Matrix

    '''
    z = np.array( e ) - np.array( t )
    z /= np.linalg.norm( z )
    x = np.cross( np.array( u ), z )
    x /= np.linalg.norm( x )
    y = np.cross( z, x )
    mat = np.identity(4)
    mat[ 0, 0:3 ] = x
    mat[ 1, 0:3 ] = y
    mat[ 2, 0:3 ] = z
    mat[ 0:3, 3 ] = -np.dot( mat[ 0:3, 0:3 ], e )
    return mat

    
def perspectiveMatrix( fovy, asp, near, far ) :
    ''' Calculate perspectiveMatrix

        Args:
            fovy : FOV (deg)
            asp  : aspect ratio
            near : near
            far  : far

        Returns:
            perspectiveMatrix

    '''
    cot = 1.0 / np.tan( np.radians( fovy / 2.0 ) )
    dz = far - near
    mat = np.zeros((4,4))
    mat[0,0] =  cot / asp
    mat[1,1] =  cot
    mat[2,2] = - (far + near)   / dz
    mat[3,2] = - 1.0
    mat[2,3] = - 2.0 * far * near / dz
    return mat


#  edited start

class OBJ :
    
    def __init__( self ) :
        self.posMat = np.identity( 4 )
        self.color  = None

#  edited end



class Obje( OBJ ) :         # <<-- edited 
    
    
    #  edited start
    
#     def __init__(self, type, positions, normals, indices ) :
#         super().__init__()
# 
#         
#         self.type = type
#         self.positions = positions
#         self.normals = normals
#         self.indices = indices
#         self.numElm  = len( indices )
#         
#         self.createVAO()
#         
#         self.color = [ 1.0, 0.0, 0.0, 1.0 ]
        
        
    def __init__( self, type      = None,
                        positions = None,
                        normals   = None,
                        indices   = None ) :
        super().__init__()
        
        self.color = [ 1.0, 0.0, 0.0, 1.0 ]
        if type is not None :
            self.setGeom( type, positions, normals, indices )
        

    def setGeom( self, type, positions, normals, indices ) :
        #print( 'setGeom Obje' )
        self.type = type
        self.positions = np.array( positions ).flatten()
        self.normals   = np.array( normals ).flatten()
        self.indices   = indices
        self.numElm    = len( indices )
        self.createVAO()
        
    #  edited end


    def draw( self, glw, posMat = None, color = None ) :  # <<-- edited
        #print('draw Obje')
        
        #  edited start
        
        if posMat is None :
            posMat = self.posMat
        else :
            posMat = np.dot( posMat, self.posMat )

        if color is None :
            if hasattr( self, 'color' ) :
                color = self.color
            else : 
                color = [ 1.0, 0.0, 0.0, 1.0 ]   
        glUniformMatrix4fv( glw.posiMatLoc, 1, GL_TRUE, posMat )
        
        #  edited end
        
        glUniform4fv( glw.colorLoc, 1, color )
        
        glBindVertexArray( self.vao )
        glDrawElements( self.type, self.numElm, GL_UNSIGNED_INT, None )     
        glBindBuffer( GL_ARRAY_BUFFER, 0 )


    def createVAO(self) :
        #print('createVAO')
        
        positions = self.positions
        normals = self.normals
        indices = self.indices
        
        self.vao = glGenVertexArrays( 1 )
        glBindVertexArray( self.vao )
        
        vbo = glGenBuffers( 1 )
        data = np.array( positions, dtype=np.float32 )
        glBindBuffer( GL_ARRAY_BUFFER, vbo ) 
        glBufferData( GL_ARRAY_BUFFER, len( positions ) * 4, data, GL_STATIC_DRAW )
        glEnableVertexAttribArray( 0 )
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, None )
 
        vbo = glGenBuffers( 1 )
        data = np.array( normals, dtype=np.float32 )
        glBindBuffer(GL_ARRAY_BUFFER, vbo) 
        glBufferData(GL_ARRAY_BUFFER, len( normals ) * 4, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray( 1 )
        glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, 0, None )

        ibo = glGenBuffers( 1 )
        data = np.array( indices, dtype=np.int32 )
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo )
        glBufferData( GL_ELEMENT_ARRAY_BUFFER, len( indices ) * 4, data, GL_STATIC_DRAW )
       
        glBindVertexArray(0)   


#  edited start

class ObjeCollection( OBJ ) :

    def __init__( self ) :
        super().__init__()
        self.objeList = []

    def addObj( self, obj ) :
        self.objeList.append( obj )
        return obj
    
    def draw( self, glw, posMat = None, color = None ) :
        #print(' draw ObjeCollection')
        
        if posMat is None :
            posMat = self.posMat
        else :
            posMat = np.dot( posMat, self.posMat )
            
        if color is None :
            color = self.color
        
        for obj in self.objeList :
            obj.draw( glw, posMat, color )


class Sphere( Obje ) :
    
    def __init__( self, R, m = 15, n = 12, img = None ) :
        #print( '__init__ Sphere' )
        super().__init__()
        the = np.linspace(   0.0        , np.pi * 2.0, m )
        phi = np.linspace( - np.pi / 2.0, np.pi / 2.0, n )
        ct, st = np.cos( the ), np.sin( the )
        cp, sp = np.cos( phi ), np.sin( phi )
        pnt, nor, idx = [], [], []
        for i in range( m ) :
            for j in range( n ) :
                x, y, z = cp[j] * ct[i], cp[j] * st[i], - sp[j]
                pnt.extend( [ R * x, R * y, R * z ] )
                nor.extend( [ x, y, z ] )
        for i in range( m - 1 ) :
            for j in range( n - 1 ) :
                k = n * i + j
                idx.extend( [ k, k + n    , k + n + 1 ] )
                idx.extend( [ k, k + n + 1, k + 1     ] )
        self.setGeom( GL_TRIANGLES, pnt, nor, idx )
        

class Coord( ObjeCollection ) :
    
    def __init__( self, L = 1.0 ) :
        super().__init__()
        for k in range( 3 ) :            
            pnt = [ ( L   if i == k+3       else 0.0 ) for i in range( 6 ) ]
            nor = [ ( 1.0 if i%3 == (k+4)%3 else 0.0 ) for i in range( 6 ) ]
            col = [ ( 1.0 if k==i or i==3   else 0.0 ) for i in range( 4 ) ]
            obj = self.addObj( Obje( GL_LINES, pnt, nor, [ 0, 1 ] ) )
            obj.color = col


#  edited end


class GLWidget( QOpenGLWidget ) :
    
    def __init__(self, parent = None ) :
        super().__init__( parent )
        self.parent = parent
        
        self.objeList = []

        self.lookAtMat = calcLookAt( [ 2.0, 3.0, 3.0 ],
                                     [ 0.0, 0.0, 0.0 ],
                                     [ 0.0, 0.0, 1.0 ] )
        self.fovy = 45.0
        self.near =  0.1
        self.far  = 20.0


    def initializeGL(self) :
        #print('initializeGL')
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable( GL_DEPTH_TEST )
        glEnable( GL_BLEND )    
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
  
        self.prog1 = self.createProgram(
            '''
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            uniform mat4 posiMat;                                        // <<-- edited
            uniform mat4 viewMat;
            void main(void){
                gl_Position = viewMat * posiMat * vec4( position, 1.0 ); // <<-- edited
            }
            '''            
            ,
            '''
            uniform vec4 color;
            void main(void){
                gl_FragColor = color;
            }
            ''' )
  
        self.posiMatLoc = glGetUniformLocation( self.prog1, 'posiMat' ) # <<-- edited
        self.colorLoc   = glGetUniformLocation( self.prog1, 'color'   ) 
        self.viewMatLoc = glGetUniformLocation( self.prog1, 'viewMat' )
  
  
        #  edited start  
  
#         xAxis = Obje( GL_LINES,
#                       [ 0.0, 0.0, 0.0,  1.0, 0.0, 0.0 ],
#                       [ 0.0, 1.0, 0.0,  0.0, 1.0, 0.0,],
#                       [ 0, 1 ] )
#         xAxis.color = [ 1.0, 0.0, 0.0, 1.0 ]
# 
#         yAxis = Obje( GL_LINES,
#                       [ 0.0, 0.0, 0.0,  0.0, 1.0, 0.0 ],
#                       [ 0.0, 0.0, 1.0,  0.0, 0.0, 1.0,],
#                       [ 0, 1 ] )
#         yAxis.color = [ 0.0, 1.0, 0.0, 1.0 ]
#         
# 
#         zAxis = Obje( GL_LINES,
#                       [ 0.0, 0.0, 0.0,  0.0, 0.0, 1.0 ],
#                       [ 1.0, 0.0, 0.0,  1.0, 0.0, 0.0,],
#                       [ 0, 1 ] )
#         zAxis.color = [ 0.0, 0.0, 1.0, 1.0 ]
#         
# 
#         obj1 = Obje( GL_TRIANGLES,
#                      [ 0.0, -0.5, 0.0, 
#                        1.0, -0.5, 0.0, 
#                        0.5, -0.5, 1.0 ],
#                      [ 0.0, 0.0, 1.0,
#                        0.0, 0.0, 1.0,
#                        0.0, 0.0, 1.0 ],
#                      [ 0, 1, 2 ] )
#         obj1.color = [ 1.0, 0.0, 0.0, 1.0 ]
# 
#         obj2 = Obje( GL_TRIANGLES,
#                      [  0.0, 0.5, 0.0, 
#                        -1.0, 0.5, 0.0, 
#                        -0.5, 0.5, 1.0 ],
#                      [ 0.0, 0.0, 1.0,
#                        0.0, 0.0, 1.0,
#                        0.0, 0.0, 1.0 ],
#                      [ 0, 1, 2 ] )
#         obj2.color = [ 0.0, 1.0, 0.0, 1.0 ]
# 
#         self.objeList = [ xAxis, yAxis, zAxis, obj1, obj2 ]
  

        coord = Coord()
        
        sphere = Sphere( 0.2 )
        sphere.color = [ 0.0, 1.0, 1.0, 1.0 ]
        sphere.posMat[ 0:3, 3 ] = [ 0.5, 0.5, 0.5 ]

        self.objeList =[ coord, sphere ] 

        #  edited end  


    def resizeGL(self, w, h) :
        #print('resizeGL')
        self.width  = w
        self.height = h
        dpr = self.parent.devicePixelRatio()
        self.viewport =[ 0, 0, int( self.width * dpr ), int( self.height * dpr ) ]
        
        
    def paintGL(self) :
        #print('paintGL')
        #        
        #-----------------------------------------------        
        #        Preparation   
        #-----------------------------------------------        
        #
        self.projMat = perspectiveMatrix( self.fovy,
                                             self.width / self.height,
                                             self.near,
                                             self.far )
        viewMat = np.dot( self.projMat, self.lookAtMat )
           
        #        
        #-----------------------------------------------        
        #        rendering      
        #-----------------------------------------------
        #        
        glUseProgram( self.prog1 )
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport( *self.viewport )
                
        glUniformMatrix4fv( self.viewMatLoc, 1, GL_TRUE, viewMat ) 
                
        for obj in self.objeList :
            obj.draw( self )


    def createProgram( self, vss, fss ) :

        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vss)
        glCompileShader(vertex_shader)

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fss)
        glCompileShader(fragment_shader)

        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glDeleteShader(vertex_shader)
        glAttachShader(program, fragment_shader)
        glDeleteShader(fragment_shader)

        glLinkProgram(program)

        return program

    def getPT( self, event ) :
        #print( 'getPT' )
        self.makeCurrent()
        X0, Y0, W0, H0 = self.viewport
        x = event.pos().x() / self.width * W0
        y = ( self.height - event.pos().y() ) / self.height * H0
        z = glReadPixelsf( x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT )[ 0, 0 ]
        p = np.array([ 2.0 * ( ( x - X0 ) / W0 ) - 1.0,
                       2.0 * ( ( y - Y0 ) / H0 ) - 1.0,
                       2.0 * z - 1.0,
                       1.0 ])
        q = np.dot( np.linalg.inv( self.projMat ), p )
        return np.array([ q[0]/q[3], q[1]/q[3], q[2]/q[3] ])
    
    
    def mousePressEvent( self, event ) :
        #print('mousePressEvent')
        self.mousePos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton :
            self.mouseBtn = 1
        elif event.button() == Qt.MouseButton.RightButton :
            self.mouseBtn = 2
            
    def mouseDoubleClickEvent( self, event ) :
        #print( 'mouseDoubleClickEvent' )
        self.selPoint = self.getPT( event )

    def mouseReleaseEvent( self, event ) :
        #print( 'mouseReleaseEvent' )
        self.mouseBtn = 0
  
    def mouseMoveEvent( self, event ) :
        #print('mouseMoveEvent')
        P0 = glGetIntegerv( GL_VIEWPORT )
        z = self.getPT( event )[2]
        dx = event.pos().x() - self.mousePos.x()
        dy = event.pos().y() - self.mousePos.y()
        self.mousePos = event.pos()
        if self.mouseBtn == 1 :
            self.lookAtMat[0,3] -= 2 * dx * z / self.width  / self.projMat[0,0] 
            self.lookAtMat[1,3] += 2 * dy * z / self.height / self.projMat[1,1]
        elif self.mouseBtn == 2 :
            if not hasattr( self, 'selPoint' ) : return
            L = 0.8 * min( self.width, self.height )
            ang1 , ang2 = dy / L , dx / L
            c1 , c2 = np.cos( ang1 ), np.cos( ang2 )
            s1 , s2 = np.sin( ang1 ), np.sin( ang2 )
            if abs( ( event.pos().x() - P0[0] )/ self.width - 0.5 ) < 0.4 :
                rotMat = np.array( [ [  c2, s2*s1, s2*c1 ],
                                     [ 0.0,   c1 ,  -s1  ],
                                     [ -s2, c2*s1, c2*c1 ] ])
                self.lookAtMat[0:3,0:3] = np.dot( rotMat, self.lookAtMat[0:3,0:3] )
                self.lookAtMat[0:3,3]   = np.dot( rotMat, self.lookAtMat[0:3,3] - self.selPoint ) \
                                        + self.selPoint
            else :
                if ( event.pos().x() - P0[0] ) / self.width > 0.5 : s1 = -s1
                rotMat = np.array( [ [  c1, -s1, 0.0 ],
                                     [  s1,  c1, 0.0 ],
                                     [ 0.0, 0.0, 1.0 ] ])
            self.lookAtMat[0:3,0:3] = np.dot( rotMat, self.lookAtMat[0:3,0:3] )
            self.lookAtMat[0:3,3]   = np.dot( rotMat, self.lookAtMat[0:3,3] - self.selPoint ) \
                                    + self.selPoint
        self.update()

    def wheelEvent( self, event ) :
        #print('wheelEvent')
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ShiftModifier :
            if event.angleDelta().y() < 0 :
                self.fovy /= 1.2
            else :
                self.fovy = self.fovy * 0.8 + 180.0 * 0.2
        else :
            if self.selPoint is None : return
            d = abs( self.selPoint[2] ) / 10.0
            if event.angleDelta().y() < 0 :
                self.lookAtMat[2,3] += d
            else :
                self.lookAtMat[2,3] -= d
        self.update()
        
        
class MainWindow(QMainWindow) :

    def __init__(self, parent=None):
        
        super( MainWindow, self ).__init__( parent )
        self.setGeometry( 100, 50, 1000, 700 )
        self.setWindowTitle( 'Main Window' )

        self.glWidget = GLWidget( self )
        self.glWidget.setGeometry( 100, 50, 800, 600 )
 
 
if __name__ == "__main__":
    app = QApplication( sys.argv )
    app.setQuitOnLastWindowClosed( True )
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec()

    