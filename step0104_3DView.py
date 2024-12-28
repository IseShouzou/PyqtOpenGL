import sys
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


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


class Obje :
    
    def __init__(self, type, positions, normals, indices ) :
        
        self.type = type
        self.positions = positions
        self.normals = normals
        self.indices = indices
        self.numElm  = len( indices )
        
        self.createVAO()
        
        self.color = [ 1.0, 0.0, 0.0, 1.0 ]
        
        
    def draw( self, paramDict ) :
        #print('draw Obje')
        
        glUniform4fv( paramDict['colorLoc'], 1, self.color )
        
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


class GLWidget( QOpenGLWidget ) :
    
    def __init__(self, parent = None ) :
        
        super().__init__( parent )
        
        self.parent = parent
        
        self.paramDict = {}
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
        glEnable( GL_ALPHA_TEST )
        glAlphaFunc( GL_EQUAL, 1 )
  
        self.prog1 = self.createProgram(
            '''
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            uniform mat4 viewMat;
            void main(void){
                gl_Position = viewMat * vec4( position, 1.0 );
            }
            '''            
            ,
            '''
            uniform vec4 color;
            void main(void){
                gl_FragColor = color;
            }
            ''' )
  
        self.paramDict[ 'colorLoc'   ] = glGetUniformLocation( self.prog1, 'color'   )
        self.paramDict[ 'viewMatLoc' ] = glGetUniformLocation( self.prog1, 'viewMat' )



        self.xAxis = Obje( GL_LINES,
                          [ 0.0, 0.0, 0.0,  1.0, 0.0, 0.0 ],
                          [ 0.0, 1.0, 0.0,  0.0, 1.0, 0.0,],
                          [ 0, 1 ] )
        self.xAxis.color = [ 1.0, 0.0, 0.0, 1.0 ]
        self.objeList.append( self.xAxis )

        self.yAxis = Obje( GL_LINES,
                          [ 0.0, 0.0, 0.0,  0.0, 1.0, 0.0 ],
                          [ 0.0, 0.0, 1.0,  0.0, 0.0, 1.0,],
                          [ 0, 1 ] )
        self.yAxis.color = [ 0.0, 1.0, 0.0, 1.0 ]
        self.objeList.append( self.yAxis )

        self.zAxis = Obje( GL_LINES,
                          [ 0.0, 0.0, 0.0,  0.0, 0.0, 1.0 ],
                          [ 1.0, 0.0, 0.0,  1.0, 0.0, 0.0,],
                          [ 0, 1 ] )
        self.zAxis.color = [ 0.0, 0.0, 1.0, 1.0 ]
        self.objeList.append( self.zAxis )



        self.obj1 = Obje( GL_TRIANGLES,
                          [ 0.0, -0.5, 0.0, 
                            1.0, -0.5, 0.0, 
                            0.5, -0.5, 1.0 ],
                          [ 0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0 ],
                          [ 0, 1, 2 ] )
        self.obj1.color = [ 1.0, 0.0, 0.0, 1.0 ]
        self.objeList.append( self.obj1 )

        self.obj2 = Obje( GL_TRIANGLES,
                          [  0.0, 0.5, 0.0, 
                            -1.0, 0.5, 0.0, 
                            -0.5, 0.5, 1.0 ],
                          [ 0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0 ],
                          [ 0, 1, 2 ] )
        self.obj2.color = [ 0.0, 1.0, 0.0, 1.0 ]
        self.objeList.append( self.obj2 )


    def resizeGL(self, w, h) :
        print('resizeGL')
        glViewport(0, 0, w, h)
        self.width  = w
        self.height = h
        
        
    def paintGL(self) :
        print( 'paintGL' )
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram( self.prog1 )
        
        self.projMatrix = perspectiveMatrix( self.fovy,
                                             self.width / self.height,
                                             self.near,
                                             self.far )
        viewMat = np.dot( self.projMatrix, self.lookAtMat )

        glUniformMatrix4fv( self.paramDict['viewMatLoc'], 1, GL_TRUE, viewMat )     

        for obj in self.objeList :
            obj.draw( self.paramDict )
        

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

    