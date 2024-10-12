
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


class GLWidget( QOpenGLWidget ) :
    
    def __init__( self, parent = None ) :
        print('GLWidget init')
        
        super().__init__( parent )
        
        self.points = np.array( [ 0.0, 0.0, 0.0, 
                                  0.5, 0.0, 0.0, 
                                  0.5, 0.5, 0.5 ] )

        self.indices = np.array( [ 0, 1, 2 ] )
        
        self.numElm  = len( self.indices )
        

    def initializeGL(self) :
        print('initializeGL')
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable( GL_DEPTH_TEST )
        glEnable( GL_ALPHA_TEST )
        glAlphaFunc( GL_EQUAL, 1 )
  
        self.prog1 = self.createProgram(
            '''
            layout(location = 0) in vec3 position;
            void main(void){
                gl_Position = vec4( position, 1.0 );
            }
            '''            
            ,
            '''          
            void main(void){
                gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
            }
            ''' )
  

        self.vao = glGenVertexArrays( 1 )      
        glBindVertexArray( self.vao )
        
        vbo = glGenBuffers(1)
        n = len( self.points )
        bytelength = n * 4
        data = np.array( self.points, dtype='float32' )
        glBindBuffer(GL_ARRAY_BUFFER, vbo ) 
        glBufferData(GL_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray( 0 )
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, None )
        
        ibo = glGenBuffers( 1 )
        n = len( self.indices )
        bytelength = n * 4
        data = np.array( self.indices, dtype='uint32' )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo )
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytelength, data, GL_STATIC_DRAW)
        
        glBindVertexArray(0)

    
    def paintGL(self) :
        print('paintGL')
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram( self.prog1 )
        glBindVertexArray( self.vao )
        glDrawElements( GL_TRIANGLES, self.numElm, GL_UNSIGNED_INT, None )     
        glBindBuffer( GL_ARRAY_BUFFER, 0 )
        
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


class MainWindow( QMainWindow ) :

    def __init__( self, parent = None ) :
        
        super().__init__()
        
        print( 'MainWindow init' )
        
        self.setGeometry( 100, 100, 500, 500 )
        self.setWindowTitle( 'Main Window' )
        self.glWidget = GLWidget()
        self.setCentralWidget( self.glWidget )

 
if __name__ == "__main__":
    app = QApplication( sys.argv )
    app.setQuitOnLastWindowClosed( True )
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec()

    