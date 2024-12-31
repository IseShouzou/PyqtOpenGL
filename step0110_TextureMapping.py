import sys
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import *    ### added

from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt

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


class OBJ :
    
    def __init__( self ) :
        self.posMat = np.identity( 4 )
        self.color  = None
       

class Obje( OBJ ) :
    
    def __init__( self, type      = None,
                        positions = None,
                        normals   = None,
                        indices   = None,
                        uvs       = None) :
        super().__init__()
        
        self.depthTestFlag  = True
        self.resetDepthFlag = False
        
        self.texFlag = False
        self.texture = None
        
        self.color = [ 1.0, 0.0, 0.0, 1.0 ]
        if type is not None :
            self.setGeom( type, positions, normals, indices, uvs )
        

    def setGeom( self, type, positions, normals, indices, uvs = None ) :
        #print( 'setGeom Obje' )
        self.type = type
        self.positions = np.array( positions ).flatten()
        self.normals   = np.array( normals ).flatten()
        self.indices   = indices
        self.numElm  = len( indices )
        if uvs is not None :
            self.uvs = np.array( uvs ).flatten()
        self.createVAO()
        
        
    def draw( self, paramDict, posMat = None, color = None ) :
        #print( 'draw Obje', self.type )

        if posMat is None :
            posMat = self.posMat
        else :
            posMat = np.dot( posMat, self.posMat )
        
        if color is None :
            if hasattr( self, 'color' ) :
                color = self.color
            else : 
                color = [ 1.0, 0.0, 0.0, 1.0 ]
        
        if paramDict[ 'prog' ] == 0 :
            glUniformMatrix4fv( paramDict[ 'posiMatLoc0' ], 1, GL_TRUE, posMat )
            
        elif paramDict[ 'prog' ] == 1 :
            glUniformMatrix4fv( paramDict[ 'posiMatLoc' ], 1, GL_TRUE, posMat )
            rgb = np.array( color )[0:3]
            glUniform3fv      ( paramDict[ 'mtlAmbiLoc' ], 1, rgb * 1.0 )
            glUniform3fv      ( paramDict[ 'mtlDiffLoc' ], 1, rgb * 0.7 )
            glUniform3fv      ( paramDict[ 'mtlSpecLoc' ], 1, rgb * 1.0 )
            glUniform1f       ( paramDict[ 'mtlShinLoc' ],    5.0       )
            glUniform1f       ( paramDict[ 'mtlAlphLoc' ],    color[3]  )
            if self.texFlag :
                glUniform1i( paramDict[ 'texFlagLoc' ], 1 )
                glUniform1i( paramDict[ 'textureLoc' ], 1 )
                glActiveTexture( GL_TEXTURE1 )
                glBindTexture( GL_TEXTURE_2D, self.texture )

            else :
                glUniform1i( paramDict[ 'texFlagLoc' ], 0 )
        
        if self.depthTestFlag :
            glEnable( GL_DEPTH_TEST )
        else :
            glDisable( GL_DEPTH_TEST )
        
        glBindVertexArray( self.vao )
        glDrawElements( self.type, self.numElm, GL_UNSIGNED_INT, None )     
        glBindBuffer( GL_ARRAY_BUFFER, 0 )

        if self.resetDepthFlag :
            glClear( GL_DEPTH_BUFFER_BIT )

        if paramDict['prog'] == 1 and self.texFlag :
            glBindTexture( GL_TEXTURE_2D, 0 )
            




    def createVAO( self ) :
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

        if hasattr( self, 'uvs' ) :
            if self.uvs is not None :
                vbo = glGenBuffers( 1 )
                data = np.array( self.uvs, dtype = np.float32 )
                glBindBuffer(GL_ARRAY_BUFFER, vbo) 
                glBufferData(GL_ARRAY_BUFFER, len( self.uvs ) * 4, data, GL_STATIC_DRAW)
                glEnableVertexAttribArray( 2 )
                glVertexAttribPointer( 2, 2, GL_FLOAT, GL_FALSE, 0, None )

        ibo = glGenBuffers( 1 )
        data = np.array( indices, dtype=np.int32 )
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, ibo )
        glBufferData( GL_ELEMENT_ARRAY_BUFFER, len( indices ) * 4, data, GL_STATIC_DRAW )
       
        glBindVertexArray(0)   


    def setTexture( self, img ) :
        #print( 'setTexture' )
        self.texFlag = True
        imgData = Image.open( img )
        w, h = imgData.size
        img  = imgData.tobytes()
        self.texture = glGenTextures( 1 )
        glBindTexture( GL_TEXTURE_2D, self.texture )
        gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, w, h, GL_RGB, GL_UNSIGNED_BYTE, imgData.tobytes() )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL) 
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE)
        glBindTexture( GL_TEXTURE_2D, 0)
        

class ObjeCollection( OBJ ) :

    def __init__( self ) :
        super().__init__()
        self.objeList = []

    def addObj( self, obj ) :
        self.objeList.append( obj )
        return obj

    def draw( self, paramDict, posMat = None, color = None ) :
        #print(' draw ObjeCollection')
        
        if posMat is None :
            posMat = self.posMat
        else :
            posMat = np.dot( posMat, self.posMat )
            
        if color is None :
            color = self.color
        
        for obj in self.objeList :
            obj.draw( paramDict, posMat, color )


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
                pnt += [ R * x, R * y, R * z ]
                nor += [ x    , y    , z     ]
        for i in range( m - 1 ) :
            for j in range( n - 1 ) :
                k = n * i + j
                idx += [ k, k + n    , k + n + 1 ]
                idx += [ k, k + n + 1, k + 1     ]
        self.setGeom( GL_TRIANGLES, pnt, nor, idx )
        
        
class Triangles( Obje ) :

    def __init__( self, pts, uvs = None ) :
        #print( '__init__ Triangles' )
        super().__init__()
        points = np.array( pts ).reshape( [ -1, 3, 3 ] )
        indices = np.arange( len( points ) * 3 )
        normals = []
        for p in points :
            v = np.cross( p[1]-p[0], p[2]-p[0] )
            normals.extend( [ v, v, v ] )
        points = points.flatten()
        self.setGeom( GL_TRIANGLES, points, normals, indices, uvs )
        

class Rectangle( Obje ) :
    
    def __init__( self, a, b, ax1 = 0 , ax2 = 1 , offset = [ 0.0, 0.0, 0.0 ] ) :
        #print( '__init__ Rectangle' )
        super().__init__()
        points = np.array( [ offset ]* 4  )
        points[ :, ax1 ] += [ -a/2.0, -a/2.0, +a/2.0, +a/2.0 ]
        points[ :, ax2 ] += [ -b/2.0, +b/2.0, -b/2.0, +b/2.0 ]
        n = np.cross( [ ( 1.0 if i == ax1 else 0.0 ) for i in range( 3 ) ] ,
                      [ ( 1.0 if i == ax2 else 0.0 ) for i in range( 3 ) ] )
        normals = np.array( [ n ] * 4  )
        indices = np.arange( 4 )
        self.setGeom( GL_TRIANGLE_STRIP, points, normals, indices )
        
        
class Circle( Obje ) :
    
    def __init__( self, r, n=20 ) :
        #print( '__init__ Circle' )
        super().__init__()
        points  = [ 0.0, 0.0, 0.0 ]
        normals = [ 0.0, 0.0, 1.0 ]
        indices = range( n + 1 )
        for t in np.linspace( 0.0,  2 * np.pi, n ) :
            points += [ r * np.cos( t ), r * np.sin( t ), 0.0 ]
            normals += [ 0.0, 0.0, 1.0 ]
        self.setGeom( GL_TRIANGLE_FAN, points, normals, indices )
 

class Cylinder( ObjeCollection ) :
    
    def __init__( self, r, h, n=20  ) :
        #print( '__init__ Cylinder' )
        super().__init__()
        self.bottom = self.addObj( Circle( r, n ) )
        self.top = self.addObj( Circle( r, n ) )
        self.top.posMat[ 2, 3 ] = h
        pnts, nors = [], []
        for t in np.linspace( 0.0,  2 * np.pi, n ) :
            c, s = np.cos( t ), np.sin( t )
            pnts += [ r * c, r * s, 0.0,  r * c, r * s, h   ]
            nors += [ c    , s    , 0.0,  c    , s    , 0.0 ]   
        idx = range( 2 * n )
        self.side = self.addObj( Obje( GL_TRIANGLE_STRIP, pnts, nors, idx ) )


class Cone( ObjeCollection ) :
    
    def __init__( self, r, h, n=20  ) :
        #print( '__init__ Cylinder' )
        super().__init__()
        rm, hm = 0.10 * r, 0.95 * h
        dm = np.sqrt( rm * rm + hm * hm )
        ar, ah = ( r - rm ) / dm, hm / dm
        
        pnts = [ 0.0, 0.0, h   ]
        nors = [ 0.0, 0.0, 1.0 ]
        for t in np.linspace( 0.0,  2 * np.pi, n ) :
            c, s = np.cos( t ), np.sin( t )
            pnts += [ rm * c, rm * s, hm ]
            nors += [ ah * c, ah * s, ar ]
        idx = range( n + 1 )    
        top = self.addObj( Obje( GL_TRIANGLE_FAN, pnts, nors, idx ) )
        
        pnts = []
        nors = []
        for t in np.linspace( 0.0,  2 * np.pi, n ) :
            c, s = np.cos( t ), np.sin( t )
            pnts += [ rm * c, rm * s, hm,   r * c , r * s , 0.0 ]
            nors += [ ah * c, ah * s, ar,   ah * c, ah * s, ar  ]   
        idx = range( 2 * n )
        side = self.addObj( Obje( GL_TRIANGLE_STRIP, pnts, nors, idx ) )
        
        bottom = self.addObj( Circle( r, n ) )
        

class ArrowCoord( ObjeCollection ) :
    
    class Arrow( ObjeCollection ) :
        def __init__( self ) :
            super().__init__()
            head = Cone    ( 0.10, 0.3 )
            tail = Cylinder( 0.05, 0.7 )
            head.posMat[ 2, 3 ] = 0.7
            self.objeList = [ head, tail ]

    def __init__( self ) :
        super().__init__()

        sphere = self.addObj( Sphere( 0.15 ) )
        sphere.color = [ 0.8, 0.8, 0.8, 1.0 ]
        xAxis = self.addObj( self.Arrow() )
        xAxis.color = [ 1.0, 0.0, 0.0, 1.0 ]
        xAxis.posMat[ 0:3, 0:3] =[ [  0.0, 0.0, 1.0 ],
                                   [  0.0, 1.0, 0.0 ],
                                   [ -1.0, 0.0, 0.0 ] ]
        yAxis = self.addObj(  self.Arrow() )
        yAxis.color = [ 0.0, 1.0, 0.0, 1.0 ]
        yAxis.posMat[ 0:3, 0:3] =[ [ 1.0,  0.0, 0.0 ],
                                   [ 0.0,  0.0, 1.0 ],
                                   [ 0.0, -1.0, 0.0 ] ]
        zAxis = self.addObj( self.Arrow() )
        zAxis.color = [ 0.0, 0.0, 1.0, 1.0 ]
        

class Coord( ObjeCollection ) :
    
    def __init__( self, L = 1.0 ) :
        super().__init__()
        for k in range( 3 ) :            
            pnt = [ ( L   if i == k+3       else 0.0 ) for i in range( 6 ) ]
            nor = [ ( 1.0 if i%3 == (k+4)%3 else 0.0 ) for i in range( 6 ) ]
            col = [ ( 1.0 if k==i or i==3   else 0.0 ) for i in range( 4 ) ]
            obj = self.addObj( Obje( GL_LINES, pnt, nor, [ 0, 1 ] ) )
            obj.color = col


class GLWidget( QOpenGLWidget ) :
    
    def __init__(self, parent = None ) :
        
        super().__init__( parent )
        
        self.parent = parent
        self.parentDevicePixelRatio = parent.devicePixelRatio()
        
        self.paramDict = {}
        self.objeList = []
       
        self.lookAtMat = calcLookAt( [ 2.0, 3.0, 3.0 ],
                                     [ 0.0, 0.0, 0.0 ],
                                     [ 0.0, 0.0, 1.0 ] )
        self.fovy = 45.0
        self.near =  0.1
        self.far  = 20.0
       
        self.lightPosi   = [  5.0, -5.0, 10.0 ]       
        self.lightAmbi   = [  0.5,  0.5,  0.5 ]
        self.lightDiff   = [  1.0,  1.0,  1.0 ]
        self.lightSpec   = [  0.2,  0.2,  0.2 ]
       
       
        self.txLightMat = np.array( [ [ 0.5, 0.0, 0.0, 0.5 ],
                                      [ 0.0, 0.5, 0.0, 0.5 ],
                                      [ 0.0, 0.0, 0.5, 0.5 ],
                                      [ 0.0, 0.0, 0.0, 1.0 ] ] )
        self.texWidth  = 4096
        self.texHeight = 4096
       
       
    def initializeGL(self) :
        #print('initializeGL')
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable( GL_DEPTH_TEST )
    
        glEnable( GL_BLEND )    
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )
        
        #
        #  Preparation of Framebuffer for Shadow Map 
        #
        self.shadowTex = glGenTextures( 1 )
        glBindTexture( GL_TEXTURE_2D, self.shadowTex )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, self.texWidth, self.texHeight,
                      0, GL_DEPTH_COMPONENT, GL_FLOAT, None )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST )        
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE )
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL ) 
        glTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE )
              
        self.shadowFB = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadowFB)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_2D, self.shadowTex, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
  
        #
        # shader for shadow map
        #
        self.prog0 = self.createProgram(
            '''
            layout(location = 0) in vec3 position;
            uniform mat4 posiMat;
            uniform mat4 ltViewMat;
            void main(void){
              gl_Position = ltViewMat * posiMat * vec4( position, 1.0 );
              gl_Position.z = gl_Position.z + 0.0005;
            }        
            ''',
            '''
            void main (void){ gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 ); }
            ''')
        
        self.paramDict['posiMatLoc0'  ] = glGetUniformLocation( self.prog0, 'posiMat'   )
        self.paramDict['ltViewMatLoc0'] = glGetUniformLocation( self.prog0, 'ltViewMat' )
        
        #
        # shader for rendering
        #
        self.prog1 = self.createProgram(
            '''
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec2 uv;
            uniform mat4 posiMat;
            uniform mat4 lookAtMat;
            uniform mat4 projMat;
            uniform mat4 viewMat;
            uniform mat4 tLtViewMat;
            varying vec4 P;
            varying vec3 N;
            varying vec2 UV;
            varying vec4 texCoord;
            void main(void){
              P = posiMat * vec4( position, 1.0 );              
              N = normalize( vec3( posiMat * vec4( normal, 0.0 ) ) );
              texCoord = tLtViewMat * P;
              gl_Position = viewMat * P;
              UV.x = uv.x;
              UV.y = 1.0 - uv.y;
            }
            '''            
            ,
            '''
            uniform vec3 eyePosi;
            uniform vec3 lightPosi;
            uniform vec3 lightAmbi;
            uniform vec3 lightDiff;
            uniform vec3 lightSpec;
            uniform vec3 mtlAmbi;
            uniform vec3 mtlDiff;
            uniform vec3 mtlSpec;
            uniform float mtlShin;
            uniform float mtlAlph;
            uniform sampler2DShadow shadowTex;
            uniform int texFlag;
            uniform sampler2D texture;
            varying vec4 P;
            varying vec3 N;
            varying vec2 UV;
            varying vec4 texCoord;
            void main(void){
              vec3 L = normalize( lightPosi * P.w - P.xyz );
              vec3 E = normalize( eyePosi   * P.w - P.xyz );
              vec3 H = normalize( L + E );
              float diffuse  = abs( dot( L, N) );
              float specular = pow( abs( dot( N, H ) ), mtlShin );
              vec3 fragColor;
              
              if ( shadow2DProj( shadowTex, texCoord ).r != 0.0 ){
                if( texFlag == 1 ){
                  fragColor = texture2D( texture, UV ).xyz;
                }else{
                  fragColor = mtlAmbi * lightAmbi
                            + mtlDiff * lightAmbi * diffuse
                            + mtlSpec * lightSpec * specular;
                }
              }else{
                if( texFlag == 1 ){
                  fragColor = texture2D( texture, UV ).xyz * 0.7;
                }else{
                  fragColor = mtlAmbi * lightAmbi
                            + mtlDiff * lightDiff * diffuse * 0.1;
                }
              }
              gl_FragColor = vec4( fragColor, mtlAlph );
            }
            ''' )
  
        self.paramDict[ 'posiMatLoc'   ] = glGetUniformLocation( self.prog1, 'posiMat'   )
        self.paramDict[ 'lookAtMatLoc' ] = glGetUniformLocation( self.prog1, 'lookAtMat' )
        self.paramDict[ 'projMatLoc'   ] = glGetUniformLocation( self.prog1, 'projMat'   )
        self.paramDict[ 'viewMatLoc'   ] = glGetUniformLocation( self.prog1, 'viewMat'   )
        self.paramDict[ 'eyePosiLoc'   ] = glGetUniformLocation( self.prog1, 'eyePosi'   )
        self.paramDict[ 'lightPosiLoc' ] = glGetUniformLocation( self.prog1, 'lightPosi' )
        
        self.paramDict[ 'lightAmbiLoc' ] = glGetUniformLocation( self.prog1, 'lightAmbi' )
        self.paramDict[ 'lightDiffLoc' ] = glGetUniformLocation( self.prog1, 'lightDiff' )
        self.paramDict[ 'lightSpecLoc' ] = glGetUniformLocation( self.prog1, 'lightSpec' )
        
        self.paramDict[ 'mtlAmbiLoc'   ] = glGetUniformLocation( self.prog1, 'mtlAmbi'   )
        self.paramDict[ 'mtlDiffLoc'   ] = glGetUniformLocation( self.prog1, 'mtlDiff'   )
        self.paramDict[ 'mtlSpecLoc'   ] = glGetUniformLocation( self.prog1, 'mtlSpec'   )
        self.paramDict[ 'mtlShinLoc'   ] = glGetUniformLocation( self.prog1, 'mtlShin'   )
        self.paramDict[ 'mtlAlphLoc'   ] = glGetUniformLocation( self.prog1, 'mtlAlph'   )
 
        self.paramDict[ 'tLtViewMatLoc'] = glGetUniformLocation( self.prog1, 'tLtViewMat')
        self.paramDict[ 'shadowTexLoc' ] = glGetUniformLocation( self.prog1, 'shadowTex' )
 
        self.paramDict[ 'texFlagLoc'   ] = glGetUniformLocation( self.prog1, 'texFlag'   )
        self.paramDict[ 'textureLoc'   ] = glGetUniformLocation( self.prog1, 'texture'   )
 
        #
        # Rendering Contents
        #
        
        self.rectangle = self.addObj( Rectangle( 5.0, 5.0, offset = [ 0.0, 0.0, -1.0] ) )
        self.rectangle.depthTestFlag = False
        self.rectangle.color = [ 0.8, 0.8, 0.8, 1.0 ]
        
        self.triangles = self.addObj( Triangles(
                                    [ -2.0, -2.0, -1.0,
                                       0.0, -2.0, -1.0,
                                       0.0,  0.0, -1.0,
                                      
                                       0.0, -2.0, -1.0,
                                       2.0, -2.0, -1.0,
                                       2.0,  0.0, -1.0,
                                      
                                      -2.0,  0.0, -1.0,
                                       0.0,  0.0, -1.0,
                                       0.0,  2.0, -1.0,
                                      
                                       0.0,  0.0, -1.0,
                                       2.0,  0.0, -1.0,
                                       2.0,  2.0, -1.0 ],
                                    
                                    [  0.0,  0.0,   0.5,  0.0,   0.5,  0.5, 
                                       0.5,  0.0,   1.0,  0.0,   1.0,  0.5, 
                                       0.0,  0.5,   0.5,  0.5,   0.5,  1.0, 
                                       0.5,  0.5,   1.0,  0.5,   1.0,  1.0  ]
                                    ) )
        
        self.triangles.setTexture( 'sky01.jpg' ) 
                
        self.arrowCoord = self.addObj( ArrowCoord() )
        


    def addObj( self, obj ) :
        self.objeList.append( obj )
        return obj


    def resizeGL( self, w, h ) :
        #print('resizeGL')
        glViewport(0, 0, w, h)
        self.width  = w
        self.height = h
        dpr = self.parentDevicePixelRatio
        self.viewport =[ 0, 0, int( self.width * dpr ), int( self.height * dpr ) ]


    def paintGL(self) :
        #print( 'paintGL' )
        
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
        eyePosi = - np.dot( self.lookAtMat[ 0:3, 0:3 ].T, self.lookAtMat[ 0:3, 3 ] )
        
        ltLkAtMat  = calcLookAt( self.lightPosi,
                                 [ 0.0, 0.0, 0.0 ],
                                 [ 0.0, 1.0, 1.0 ] )
        ltProjMatprojMat = perspectiveMatrix( 45.0, 1.0, 1.0, 20.0 )
        ltViewMat = np.dot( ltProjMatprojMat, ltLkAtMat )
        tLtViewMat = np.dot( self.txLightMat, ltViewMat )
 
        #        
        #-----------------------------------------------        
        #        1st step  :  create shadow map    
        #-----------------------------------------------        
        #
        glUseProgram( self.prog0 )
        self.paramDict[ 'prog' ] = 0

        glBindFramebuffer( GL_FRAMEBUFFER, self.shadowFB )
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glViewport( 0, 0, self.texWidth, self.texHeight )
        glUniformMatrix4fv( self.paramDict[ 'ltViewMatLoc0' ], 1, GL_TRUE, ltViewMat )
        
        for obj in self.objeList :
            obj.draw( self.paramDict )
        
        glBindFramebuffer( GL_FRAMEBUFFER, 0 )
        
        #        
        #-----------------------------------------------        
        #        2nd step  :  rendering      
        #-----------------------------------------------
        #        
        self.makeCurrent()
        glUseProgram( self.prog1 )
        self.paramDict[ 'prog' ] = 1
        
        glActiveTexture( GL_TEXTURE0 )
        glBindTexture( GL_TEXTURE_2D, self.shadowTex )
        
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glViewport( *self.viewport )
                
        glUniformMatrix4fv( self.paramDict[ 'lookAtMatLoc' ], 1, GL_TRUE, self.lookAtMat )     
        glUniformMatrix4fv( self.paramDict[ 'projMatLoc'   ], 1, GL_TRUE, self.projMat   )     
        glUniformMatrix4fv( self.paramDict[ 'viewMatLoc'   ], 1, GL_TRUE, viewMat        )     
        glUniform3fv      ( self.paramDict[ 'eyePosiLoc'   ], 1,          eyePosi        )
        glUniform3fv      ( self.paramDict[ 'lightPosiLoc' ], 1,          self.lightPosi )

        glUniform3fv      ( self.paramDict['lightAmbiLoc'  ], 1,          self.lightAmbi )
        glUniform3fv      ( self.paramDict['lightDiffLoc'  ], 1,          self.lightDiff )
        glUniform3fv      ( self.paramDict['lightSpecLoc'  ], 1,          self.lightSpec )

        glUniformMatrix4fv( self.paramDict[ 'tLtViewMatLoc'], 1, GL_TRUE, tLtViewMat     )     
        glUniform1i       ( self.paramDict['shadowTexLoc'  ],             0              )

        for obj in self.objeList :
            obj.draw( self.paramDict )
        
        glUseProgram(0)


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

    