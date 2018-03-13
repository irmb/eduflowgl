#include "Visualizer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>

#include "Utility/NacaProfile.h"

GLFWwindow* Visualizer::gWindow = NULL;
tdogl::Program* Visualizer::gProgram = NULL;

uint Visualizer::nx = 0;
uint Visualizer::ny = 0;

float Visualizer::pxPerVertex = 0;

uint Visualizer::vertexArrayID   = 0;
uint Visualizer::vertexBufferID  = 0;
uint Visualizer::elementBufferID = 0;

std::vector<float> Visualizer::vertices;
std::vector<uint>  Visualizer::elements;

lbmSolverPtr Visualizer::solver = nullptr;

char Visualizer::lbModel = 'b';

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

static void APIENTRY openglCallbackFunction(
  GLenum source,
  GLenum type,
  GLuint id,
  GLenum severity,
  GLsizei length,
  const GLchar* message,
  const void* userParam
){
  (void)source; (void)type; (void)id; 
  (void)severity; (void)length; (void)userParam;
  fprintf(stderr, "%s\n", message);
  if (severity==GL_DEBUG_SEVERITY_HIGH) {
    fprintf(stderr, "Aborting...\n");
    abort();
  }
}


void OnError(int errorCode, const char* msg) {
    throw std::runtime_error(msg);
}

void Visualizer::initialize(uint nx, uint ny, float pxPerVertex, uint timeStepsPerFrame, lbmSolverPtr solver)
{
    Visualizer::nx          = nx;
    Visualizer::ny          = ny;
    Visualizer::pxPerVertex = pxPerVertex;
    Visualizer::solver      = solver;
    
    Visualizer::timeStepsPerFrame = timeStepsPerFrame;

    Visualizer::stopWatch   = std::make_shared<StopWatch>();

    //////////////////////////////////////////////////////////////////////////

    // initialise GLFW
    glfwSetErrorCallback(OnError);
    if(!glfwInit())
        throw std::runtime_error("glfwInit failed");
    
    // open a window with GLFW
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    Visualizer::gWindow = glfwCreateWindow(pxPerVertex*nx, pxPerVertex*ny, "iRMB - Interactive LBM Solver with CUDA and OpenGL", NULL, NULL);
    if(!Visualizer::gWindow)
        throw std::runtime_error("glfwCreateWindow failed. Can your hardware handle OpenGL 3.2?");

    // GLFW settings
    glfwMakeContextCurrent(Visualizer::gWindow);
    
    //glfwSetKeyCallback(Visualizer::gWindow, key_callback);
    //glfwSetCursorPosCallback(Visualizer::gWindow, mouse_position_callback);
    //glfwSetMouseButtonCallback(Visualizer::gWindow, mouse_button_callback);

    // initialise GLEW
    glewExperimental = GL_TRUE; //stops glew crashing on OSX :-/
    if(glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");

    // print out some info about the graphics drivers
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

#ifdef _DEBUG
    // Enable the debug callback
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(openglCallbackFunction, nullptr);
    glDebugMessageControl( GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, true );
#endif
    //////////////////////////////////////////////////////////////////////////
}

void Visualizer::installShaders()
{
    std::vector<tdogl::Shader> shaders;
#ifdef __APPLE__
    shaders.push_back(tdogl::Shader::shaderFromFile("../application/InteractiveOpenGL/resources/vertex-shader.txt", GL_VERTEX_SHADER));
    shaders.push_back(tdogl::Shader::shaderFromFile("../application/InteractiveOpenGL/resources/fragment-shader.txt", GL_FRAGMENT_SHADER));
#else
    shaders.push_back(tdogl::Shader::shaderFromFile("../../../application/InteractiveOpenGL/resources/vertex-shader.txt", GL_VERTEX_SHADER));
    shaders.push_back(tdogl::Shader::shaderFromFile("../../../application/InteractiveOpenGL/resources/fragment-shader.txt", GL_FRAGMENT_SHADER));
#endif
    gProgram = new tdogl::Program(shaders);
}

void Visualizer::generateVertices()
{
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    glGenBuffers(1, &vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);

    for( uint yIdx = 0; yIdx < ny; yIdx++ ){
        for( uint xIdx = 0; xIdx < nx; xIdx++ ){

            GLfloat x = -1.0f + (2.0f/float(nx-1)) * float(xIdx);
            GLfloat y = -1.0f + (2.0f/float(ny-1)) * float(yIdx);

            vertices.push_back( x );
            vertices.push_back( y );
            vertices.push_back( 0.0f );

            vertices.push_back( 0.0f );
            vertices.push_back( 0.0f );
            vertices.push_back( 0.0f );
        }
    }

    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(gProgram->attrib("vert"));
    glVertexAttribPointer(gProgram->attrib("vert"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, NULL);

    glEnableVertexAttribArray(gProgram->attrib("color"));
    glVertexAttribPointer(gProgram->attrib("color"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (char*)(sizeof(float) * 3));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Visualizer::generateElements()
{
    glGenBuffers(1, &elementBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferID);

    for( uint yIdx = 0; yIdx < ny-1; yIdx++ ){
        for( uint xIdx = 0; xIdx < nx-1; xIdx++ ){

            elements.push_back( solver->c2i( xIdx + 0, yIdx + 0 ) );
            elements.push_back( solver->c2i( xIdx + 1, yIdx + 0 ) );
            elements.push_back( solver->c2i( xIdx + 0, yIdx + 1 ) );

            elements.push_back( solver->c2i( xIdx + 1, yIdx + 1 ) );
            elements.push_back( solver->c2i( xIdx + 0, yIdx + 1 ) );
            elements.push_back( solver->c2i( xIdx + 1, yIdx + 0 ) );
        }
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(uint), elements.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Visualizer::run()
{
    while(!glfwWindowShouldClose(gWindow)){
        // process pending events
        glfwPollEvents();

        // draw one frame
        displayCall();
    }

    glfwTerminate();
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

    solver->postProcessing( postProcessingType );

    //////////////////////////////////////////////////////////////////////////

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(gProgram->object());

    glBindVertexArray(vertexArrayID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferID);

    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, nullptr);
    //glDrawElements(GL_QUADS, elements.size(), GL_UNSIGNED_INT, nullptr);

    

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);

    glfwSwapBuffers(gWindow);
}

void Visualizer::click(int button, int updown, int x, int y)
{
    //clicked = !clicked;

    //if( button == GLUT_RIGHT_BUTTON && updown == GLUT_DOWN )
    //    delelteGeo = true;
    //else
    //    delelteGeo = false;

    //std::cout << "Clicked at ( " << x << ", " << y << " )" << std::endl;

    //int xIdx = ( float(nx) - 8.0f/pxPerVertex ) * float(                  x ) / float( nx*pxPerVertex ) ;
    //int yIdx = ( float(ny) - 8.0f/pxPerVertex ) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;

    //if(clicked) solver->setGeo( xIdx, yIdx, delelteGeo?0:1 );

    //xIdxLast = xIdx;
    //yIdxLast = yIdx;
}

void Visualizer::motion(int x, int y)
{
    //std::cout << "Motioned at ( " << x << ", " << y << " )" << std::endl;

    //int xIdx = ( float(nx) - 8.0f/pxPerVertex ) * float(                  x ) / float( nx*pxPerVertex ) ;
    //int yIdx = ( float(ny) - 8.0f/pxPerVertex ) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;
    //
    //std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    //if( xIdx <  0 || xIdx > 1000000 ) xIdx = 0;
    //if( yIdx <  0 || yIdx > 1000000 ) yIdx = 0;
    //if( xIdx >= nx - 2 )              xIdx = nx - 2;
    //if( yIdx >= ny - 2 )              yIdx = ny - 2;

    //std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    //solver->setGeo( xIdxLast, yIdxLast, xIdx, yIdx, delelteGeo?0:1 );

    //geoModified = true;

    //xIdxLast = xIdx;
    //yIdxLast = yIdx;

    //Visualizer::solver->scaleColorMap();
}

void Visualizer::keyboard(unsigned char key, int x, int y)
{
  //  int R;
  //  switch (key)
  //  {

  //      case 'h':
  //          std::cout << "================================================================================" << std::endl;
  //          std::cout << "          H e l p :" << std::endl;
  //          std::cout << "================================================================================" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "Left Mouse:      Draw solid nodes" << std::endl;
  //          std::cout << "Right Mouse:     Delete solid nodes" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "b:  toggle BGK and Central Moment methods" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "g:  Reset solid nodes" << std::endl;
  //          std::cout << "r:  Reset flow field" << std::endl;
  //          std::cout << "c:  Draw cylinder" << std::endl;
  //          std::cout << "p:  Show pressure field" << std::endl;
  //          std::cout << "v:  Show velocity field" << std::endl;
  //          std::cout << "s:  Scale color map" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "n:  Show performance" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "*:  increase viscosity by factor 10" << std::endl;
  //          std::cout << "/:  reduce viscosity by factor 10" << std::endl;
  //          std::cout << "+:  increase velocity by 0.01 dx/dt" << std::endl;
  //          std::cout << "-:  reduce velocity by 0.01 dx/dt" << std::endl;
  //          std::cout << "" << std::endl;
  //          std::cout << "8:  increase time steps per frame by one" << std::endl;
  //          std::cout << "2:  decrease time steps per frame by one" << std::endl;
		//	std::cout << "4:  turn velocity clockwise" << std::endl;
		//	std::cout << "6:  turn velocity anti-clockwise" << std::endl;
  //          std::cout << "================================================================================" << std::endl;
  //          break;

  //      case 'b':
  //          if( solver->getLBModel() == 'b' ){
  //              solver->setLBModel('c');
  //              std::cout << "Using central moment method!" << std::endl;
  //          }
  //          else{
  //              solver->setLBModel('b');
  //              std::cout << "Using BGK!" << std::endl;
  //          }
  //          break;

  //      case 'w':
  //          if( solver->getGeoMode() == 'b' ){
  //              solver->setGeoMode('w');
  //          }
  //          else{
  //              solver->setGeoMode('b');
  //          }
  //          break;

  //      case 'v':
  //          std::cout << "Show Velocity!" << std::endl;
  //          postProcessingType = 'v';
  //          solver->scaleColorMap();
  //          break;

  //      case 'p':
  //          std::cout << "Show Pressure!" << std::endl;
  //          postProcessingType = 'p';
  //          solver->scaleColorMap();
  //          break;

  //      case 'r':
  //          std::cout << "Initialize Distributions!" << std::endl;
  //          solver->initializeDistributions();
  //          break;

  //      case 'g':
  //          std::cout << "Initialize Distributions!" << std::endl;
  //          solver->initializeGeo();
  //          break;

  //      case 's':
  //          std::cout << "Scale ColorMap!" << std::endl;
  //          solver->scaleColorMap();
  //          break;

  //      case 'n':
  //          std::cout << "Performance: " << std::endl;
  //          std::cout << "    " << nups << " NUPS" << std::endl;
  //          std::cout << "    " << fps  << " FPS"  << std::endl;
  //          break;

  //      case '*':
  //          Visualizer::solver->setNu( 10.0f * Visualizer::solver->getNu() );
  //          std::cout << "Viscosity = " << Visualizer::solver->getNu() << std::endl;
  //          break;

  //      case '/':
  //          Visualizer::solver->setNu( 0.1f * Visualizer::solver->getNu() );
  //          std::cout << "Viscosity = " << Visualizer::solver->getNu() << std::endl;
  //          break;

  //      case '+':
  //          Visualizer::solver->setSpeed( Visualizer::solver->getSpeed() + 0.001f );
  //          std::cout << "U = " << Visualizer::solver->getU() << " V = " << Visualizer::solver->getV() << std::endl;
  //          break;

  //      case '-':
  //          Visualizer::solver->setSpeed( Visualizer::solver->getSpeed() - 0.001f );
		//	std::cout << "U = " << Visualizer::solver->getU() << " V = " << Visualizer::solver->getV() << std::endl;
  //          break;

  //      case '8':
  //          Visualizer::timeStepsPerFrame++;
  //          std::cout << "dt / frame = " << Visualizer::timeStepsPerFrame << std::endl;
  //          break;

  //      case '2':
  //          Visualizer::timeStepsPerFrame--;
  //          if( Visualizer::timeStepsPerFrame < 1 ) Visualizer::timeStepsPerFrame = 1;
  //          std::cout << "dt / frame = " << Visualizer::timeStepsPerFrame << std::endl;
  //          break;

		//case '4':
		//	Visualizer::solver->setAlpha(Visualizer::solver->getAlpha() + 2.f*3.141592654f/360.f);
		//	Visualizer::solver->setSpeed(Visualizer::solver->getSpeed());
		//	std::cout << "alpha = " << Visualizer::solver->getAlpha() << std::endl;
		//	break;

		//case '6':
		//	Visualizer::solver->setAlpha(Visualizer::solver->getAlpha() - 2.f*3.141592654f / 360.f);
		//	Visualizer::solver->setSpeed(Visualizer::solver->getSpeed());
		//	std::cout << "alpha = " << Visualizer::solver->getAlpha() << std::endl;
		//	break;

  //      case 'c':
  //          R = Visualizer::ny/4;
  //          for( int x = -R; x <= R; x++ ){
  //              for( int y = -R; y <= R; y++ ){
  //                  if( sqrt( x * x + y * y ) > R ) continue;
  //                  solver->setGeo(2*R + x, 2*R + y,1);
  //              }
  //          }
  //          break;

  //      case 'a':
  //          for( uint idx = 0; idx < nacaProfilePoints; idx++  ){

  //              float x1 = Visualizer::ny/8 + Visualizer::ny/2 * nacaProfile[(idx  )%nacaProfilePoints][0];
  //              float y1 = Visualizer::ny/2 + Visualizer::ny/2 * nacaProfile[(idx  )%nacaProfilePoints][1];
  //              float x2 = Visualizer::ny/8 + Visualizer::ny/2 * nacaProfile[(idx+1)%nacaProfilePoints][0];
  //              float y2 = Visualizer::ny/2 + Visualizer::ny/2 * nacaProfile[(idx+1)%nacaProfilePoints][1];

  //              solver->setGeo( x1, y1, x2, y2, 1 );
  //          }
  //          break;

  //      default:
  //          break;
  //  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint Visualizer::getVertexBufferID()
{
    return vertexBufferID;
}
