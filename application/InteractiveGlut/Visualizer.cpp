#include "Visualizer.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>

uint Visualizer::nx = 0;
uint Visualizer::ny = 0;

uint Visualizer::pxPerVertex = 0;

uint Visualizer::vertexBufferID  = 0;
uint Visualizer::elementBufferID = 0;

std::vector<float> Visualizer::vertices;
std::vector<uint>  Visualizer::elements;

lbmSolverPtr Visualizer::solver = nullptr;

bool Visualizer::clicked     = false;
bool Visualizer::geoModified = false;
bool Visualizer::delelteGeo  = false;

uint Visualizer::xIdxLast = 0;
uint Visualizer::yIdxLast = 0;

char Visualizer::postProcessingType = 'v';

uint Visualizer::timeStepsPerFrame = 100;

StopWatchPtr Visualizer::stopWatch = nullptr;

double Visualizer::nups = 0.0; 
double Visualizer::fps  = 0.0; 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Visualizer::initialize(int argc, char *argv[], uint nx, uint ny, uint pxPerVertex, uint timeStepsPerFrame, lbmSolverPtr solver)
{
    Visualizer::nx          = nx;
    Visualizer::ny          = ny;
    Visualizer::pxPerVertex = pxPerVertex;
    Visualizer::solver      = solver;
    
    Visualizer::timeStepsPerFrame = timeStepsPerFrame;

    Visualizer::stopWatch   = std::make_shared<StopWatch>();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize( pxPerVertex*nx + 16, pxPerVertex*ny + 16 );
    glutInitWindowPosition(100, 100);
    glutCreateWindow("iRMB!");
    glutDisplayFunc(Visualizer::displayCall);
    glutMouseFunc(Visualizer::click);
    glutMotionFunc(Visualizer::motion);
    glutKeyboardFunc(Visualizer::keyboard);

    glewInit();

    std::cout << glutGet(GLUT_WINDOW_WIDTH)  << std::endl;
    std::cout << glutGet(GLUT_WINDOW_HEIGHT) << std::endl;
}

void Visualizer::installShaders()
{
    GLuint vertexShaderID   = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    const char * vertexShaderCode = 
        "#version 430\r\n"
        ""
        "in layout(location=0) vec2 inPosition;"
        "in layout(location=1) vec3 inVertexColor;"
        "out vec3 outVertexColor;"
        ""
        "void main()"
        "{"
        "   gl_Position = vec4(inPosition, 0.0, 1.0);"
        "   outVertexColor = inVertexColor;"
        "}";

    const char * fragmentShaderCode = 
        "#version 430\r\n"
        ""
        "in vec3 outVertexColor;"
        "out vec4 fragementColor;"
        ""
        "void main()"
        "{"
        "   fragementColor = vec4(outVertexColor, 1.0);"
        "}";

    const char * adapter [1];

    adapter[0] = vertexShaderCode;
    glShaderSource(vertexShaderID,   1, adapter, 0);
    adapter[0] = fragmentShaderCode;
    glShaderSource(fragmentShaderID, 1, adapter, 0);

    glCompileShader(vertexShaderID);
    glCompileShader(fragmentShaderID);

    GLuint programID = glCreateProgram();

    glAttachShader(programID, vertexShaderID);
    glAttachShader(programID, fragmentShaderID);

    glLinkProgram(programID);

    glUseProgram(programID);
}

void Visualizer::generateVertices()
{
    for( uint yIdx = 0; yIdx < ny; yIdx++ ){
        for( uint xIdx = 0; xIdx < nx; xIdx++ ){

            GLfloat x = -1.0f + (2.0f/float(nx-1)) * float(xIdx);
            GLfloat y = -1.0f + (2.0f/float(ny-1)) * float(yIdx);

            vertices.push_back( x );
            vertices.push_back( y );

            vertices.push_back( 0.0f );
            vertices.push_back( 0.0f );
            vertices.push_back( 0.0f );
        }
    }

    glGenBuffers(1, &vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (char*)(sizeof(float) * 2));
}

void Visualizer::generateElements()
{
    for( uint yIdx = 0; yIdx < ny-1; yIdx++ ){
        for( uint xIdx = 0; xIdx < nx-1; xIdx++ ){

            elements.push_back( solver->c2i( xIdx + 0, yIdx + 0 ) );
            elements.push_back( solver->c2i( xIdx + 1, yIdx + 0 ) );
            elements.push_back( solver->c2i( xIdx + 1, yIdx + 1 ) );
            elements.push_back( solver->c2i( xIdx + 0, yIdx + 1 ) );
        }
    }

    glGenBuffers(1, &elementBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(GLuint), elements.data(), GL_STATIC_DRAW);
}

void Visualizer::run()
{
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Visualizer::displayCall()
{
    for( int i = 0; i < timeStepsPerFrame; i++ )
        solver->collision();

    nups = ulint(nx) *ulint(ny) * ulint(timeStepsPerFrame) * 1000.0 / stopWatch->getElapsedMilliSeconds();
    fps  = 1000.0 / stopWatch->getElapsedMilliSeconds();
    stopWatch->reset();

    //////////////////////////////////////////////////////////////////////////

    solver->postProcessing( postProcessingType );

    glDrawElements(GL_QUADS, elements.size(), GL_UNSIGNED_INT, 0);

    glutSwapBuffers();

    glutPostRedisplay();
}

void Visualizer::click(int button, int updown, int x, int y)
{
    clicked = !clicked;

    if( button == GLUT_RIGHT_BUTTON && updown == GLUT_DOWN )
        delelteGeo = true;
    else
        delelteGeo = false;

    std::cout << "Clicked at ( " << x << ", " << y << " )" << std::endl;

    int xIdx = ( float(nx) - 8.0f/pxPerVertex ) * float(                  x ) / float( nx*pxPerVertex ) ;
    int yIdx = ( float(ny) - 8.0f/pxPerVertex ) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;

    if(clicked) solver->setGeo( xIdx, yIdx, delelteGeo?0:1 );

    xIdxLast = xIdx;
    yIdxLast = yIdx;
}

void Visualizer::motion(int x, int y)
{
    //std::cout << "Motioned at ( " << x << ", " << y << " )" << std::endl;

    int xIdx = ( float(nx) - 8.0f/pxPerVertex ) * float(                  x ) / float( nx*pxPerVertex ) ;
    int yIdx = ( float(ny) - 8.0f/pxPerVertex ) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;
    
    std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    if( xIdx <  0 || xIdx > 1000000 ) xIdx = 0;
    if( yIdx <  0 || yIdx > 1000000 ) yIdx = 0;
    if( xIdx >= nx - 2 )              xIdx = nx - 2;
    if( yIdx >= ny - 2 )              yIdx = ny - 2;

    std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    int dxIdx = xIdx - xIdxLast;
    int dyIdx = yIdx - yIdxLast;

    if( abs(dxIdx) >= abs(dyIdx) ){
        for( uint idx = 0; idx < abs(dxIdx); idx++ ){
    
            float xInc = ( dxIdx != 0 )?( float(dxIdx) / float( abs(dxIdx) ) ):(0);
            float yInc = ( dxIdx != 0 )?( float(dyIdx) / float( abs(dxIdx) ) ):(0);
            
            int x = int(xIdxLast) + float(idx) * xInc;
            int y = int(yIdxLast) + float(idx) * yInc;

            solver->setGeo(x,y, delelteGeo?0:1);
        }
    }else{
        for( uint idx = 0; idx < abs(dyIdx); idx++ ){
    
            float xInc = ( dyIdx != 0 )?( float(dxIdx) / float( abs(dyIdx) ) ):(0);
            float yInc = ( dyIdx != 0 )?( float(dyIdx) / float( abs(dyIdx) ) ):(0);

            int x = int(xIdxLast) + float(idx) * xInc;
            int y = int(yIdxLast) + float(idx) * yInc;

            solver->setGeo(x,y, delelteGeo?0:1);
        }
    }

    geoModified = true;

    xIdxLast = xIdx;
    yIdxLast = yIdx;

    Visualizer::solver->scaleColorMap();
}

void Visualizer::keyboard(unsigned char key, int x, int y)
{
    int R;
    switch (key)
    {
        case 'v':
            std::cout << "Show Velocity" << std::endl;
            postProcessingType = 'v';
            solver->scaleColorMap();
            break;

        case 'p':
            std::cout << "Show Pressure" << std::endl;
            postProcessingType = 'p';
            solver->scaleColorMap();
            break;

        case 'r':
            std::cout << "Initialize Distributions" << std::endl;
            solver->initializeDistributions();
            break;

        case 's':
            std::cout << "Initialize Distributions" << std::endl;
            solver->scaleColorMap();
            break;

        case 'n':
            std::cout << "Performance: " << std::endl;
            std::cout << "    " << nups << " NUPS" << std::endl;
            std::cout << "    " << fps  << " FPS"  << std::endl;
            break;

        case '*':
            Visualizer::solver->setNu( 10.0f * Visualizer::solver->getNu() );
            std::cout << "Viscosity = " << Visualizer::solver->getNu() << std::endl;
            break;

        case '/':
            Visualizer::solver->setNu( 0.1f * Visualizer::solver->getNu() );
            std::cout << "Viscosity = " << Visualizer::solver->getNu() << std::endl;
            break;

        case '+':
            Visualizer::solver->setU( Visualizer::solver->getU() + 0.001f );
            std::cout << "U = " << Visualizer::solver->getU() << std::endl;
            break;

        case '-':
            Visualizer::solver->setU( Visualizer::solver->getU() - 0.001f );
            std::cout << "U = " << Visualizer::solver->getU() << std::endl;
            break;

        case '8':
            Visualizer::timeStepsPerFrame++;
            std::cout << "dt / frame = " << Visualizer::timeStepsPerFrame << std::endl;
            break;

        case '2':
            Visualizer::timeStepsPerFrame--;
            if( Visualizer::timeStepsPerFrame < 1 ) Visualizer::timeStepsPerFrame = 1;
            std::cout << "dt / frame = " << Visualizer::timeStepsPerFrame << std::endl;
            break;

        case 'c':
            R = Visualizer::ny/4;
            for( int x = -R; x <= R; x++ ){
                for( int y = -R; y <= R; y++ ){
                    if( sqrt( x * x + y * y ) > R ) continue;
                    solver->setGeo(2*R + x, 2*R + y,1);
                }
            }
            break;

        default:
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint Visualizer::getVertexBufferID()
{
    return vertexBufferID;
}
