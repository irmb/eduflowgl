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

StopWatchPtr Visualizer::stopWatch = nullptr;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Visualizer::initialize(int argc, char *argv[], uint nx, uint ny, uint pxPerVertex, lbmSolverPtr solver)
{
    Visualizer::nx          = nx;
    Visualizer::ny          = ny;
    Visualizer::pxPerVertex = pxPerVertex;
    Visualizer::solver      = solver;
    
    Visualizer::stopWatch   = std::make_shared<StopWatch>();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(pxPerVertex*nx+16, pxPerVertex*ny+16);
    glutInitWindowPosition(300, 200);
    glutCreateWindow("iRMB!");
    glutDisplayFunc(Visualizer::displayCall);
    glutMouseFunc(Visualizer::click);
    glutMotionFunc(Visualizer::motion);
    glutKeyboardFunc(Visualizer::keyboard);

    glewInit();
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

            GLfloat x = -1.0f + 2.0f/float(nx-1) * float(xIdx);
            GLfloat y = -1.0f + 2.0f/float(ny-1) * float(yIdx);

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

    uint timeStepsPerVisualisation = 100;

    for( int i = 0; i < timeStepsPerVisualisation; i++ )
        solver->collision();

    std::cout << ulint(nx) *ulint(ny) * ulint(timeStepsPerVisualisation) * 1000.0 / stopWatch->getElapsedMilliSeconds() << std::endl;
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

    uint xIdx =      nx * ( float(x) / float( nx*pxPerVertex + 16 ) ) ;
    uint yIdx = ny - ny * ( float(y) / float( ny*pxPerVertex + 16 ) ) ;

    yIdx -= 3;

    xIdxLast = xIdx;
    yIdxLast = yIdx;
}

void Visualizer::motion(int x, int y)
{
    //std::cout << "Motioned at ( " << x << ", " << y << " )" << std::endl;

    uint xIdx =      nx * ( float(x) / float( nx*pxPerVertex + 16 ) ) ;
    uint yIdx = ny - ny * ( float(y) / float( ny*pxPerVertex + 16 ) ) ;

    yIdx -= 3;

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
}

void Visualizer::keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'v':
            std::cout << "Post Process Velocity" << std::endl;
            postProcessingType = 'v';
            break;

        case 'p':
            std::cout << "Post Process Pressure" << std::endl;
            postProcessingType = 'p';
            break;

        case 'r':
            std::cout << "Initialize Distributions" << std::endl;
            solver->initializeDistributions();
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
