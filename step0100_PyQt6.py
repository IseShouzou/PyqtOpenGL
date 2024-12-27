
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

import sys
import numpy as np


class GLWidget( QOpenGLWidget ) :
    
    def __init__(self, parent=None ) :
        super().__init__( parent )
        #print( 'GLWidget init' )
        self.parent = parent

    def initializeGL( self ) :
        #print( 'initializeGL' )     
        glClearColor( 0.0, 0.0, 0.0, 1.0 )
        glEnable( GL_DEPTH_TEST )
  
    def resizeGL( self, w, h) :
        #print( 'resizeGL' )
        glViewport( 0, 0, w, h )
        self.width  = w
        self.height = h
        
    def paintGL(self) :
        #print('paintGL')
        
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glBegin( GL_POLYGON )
        glColor3f( 1.0, 0.0, 0.0 )
        glVertex3f( 0.0, 0.8, 0.0 )
        glColor3f( 0.0, 1.0, 0.0 )
        glVertex3f( 0.8, -0.8, 0.0 )
        glColor3f( 0.0, 0.0, 1.0 )
        glVertex3f( -0.8, -0.8, 0.0 )
        glEnd()
        

class MainWindow( QMainWindow ) :

    def __init__( self, parent=None ) :
        
        super( MainWindow, self ).__init__( parent )
        self.setGeometry( 100, 50, 1000, 700 )
        self.setWindowTitle( 'Main Window' )

        self.glWidget = GLWidget( self )
        self.glWidget.setGeometry( 100, 50, 800, 600 )
       
if __name__ == '__main__' :
    app = QApplication( sys.argv )
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec()

    
