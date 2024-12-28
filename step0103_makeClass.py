import sys
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

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
            void main(void){
                gl_Position = vec4( position, 1.0 );
            }
            '''            
            ,
            '''
            uniform vec4 color;
            void main(void){
                gl_FragColor = color;
            }
            ''' )
  
        self.paramDict[ 'colorLoc' ] = glGetUniformLocation( self.prog1, 'color' )
        
        self.obj1 = Obje( GL_TRIANGLES,
                          [ 0.0, 0.0, 0.0, 
                            1.0, 0.0, 0.0, 
                            0.5, 1.0, 0.0 ],
                          [ 0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0,
                            0.0, 0.0, 1.0 ],
                          [ 0, 1, 2 ] )
        self.obj1.color = [ 1.0, 0.0, 0.0, 1.0 ]
        self.objeList.append( self.obj1 )

        self.obj2 = Obje( GL_TRIANGLES,
                          [  0.0, 0.0, 0.0, 
                            -1.0, 0.0, 0.0, 
                            -0.5, 1.0, 0.0 ],
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
        print('paintGL')
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram( self.prog1 )
        
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

    