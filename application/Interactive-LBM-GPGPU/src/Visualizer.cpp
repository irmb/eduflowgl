#include "visualizer.h"
#include "Command.h"
#include "ReadSolidGeometryCommand.h"
#include "writeFlowFieldToVTKCommand.h"


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

#include "Utility/NacaProfile.h"
#include "Utility/CarGeometry.h"




#include <vector>
#include <stdexcept>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <fstream>
#include <thrust/device_vector.h>


#include <ctime>

std::unique_ptr<Command> command;
std::vector<std::unique_ptr<Command>> history;
    int currentCommand = -1;


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

Visualizer::Visualizer(uint nx, uint ny, float pxPerVertex, uint timeStepsPerFrame, LBMSolverPtr solver)
{
    

    this->nx          = nx;
    this->ny          = ny;
    //Visualizer::lref          = lref;
    this->pxPerVertex = pxPerVertex;
    this->solver      = solver;
    
    this->timeStepsPerFrame = timeStepsPerFrame;

    this->lbModel = 'b';
    this->postProcessingType = 'v';

    this->isMouseButtonPressed = false;
    this->geoModified          = false;
    this->delelteGeo           = false;

    this->stopWatch   = std::make_shared<StopWatch>();

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

    this->gWindow = glfwCreateWindow(pxPerVertex*nx, pxPerVertex*ny, "edu::flow", NULL, NULL);
    if(!this->gWindow)
        throw std::runtime_error("glfwCreateWindow failed. Can your hardware handle OpenGL 3.2?");

    // GLFW settings
    glfwMakeContextCurrent(this->gWindow);

    glfwSetKeyCallback        (this->gWindow, Visualizer::keyboardCallbackWrapper);
    glfwSetCursorPosCallback  (this->gWindow, Visualizer::mouseMotionCallbackWrapper);
    glfwSetMouseButtonCallback(this->gWindow, Visualizer::mouseButtonCallbackWrapper);
    
    glfwSetWindowUserPointer( this->gWindow, this );

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
    
    std::string resourcePath = __FILE__;

#ifdef _WIN32
			resourcePath = resourcePath.substr(0, resourcePath.find_last_of('\\') + 1);
#else
			resourcePath = resourcePath.substr(0, resourcePath.find_last_of('/') + 1);
#endif

    shaders.push_back(tdogl::Shader::shaderFromFile(resourcePath + "../resources/vertex-shader.txt", GL_VERTEX_SHADER));
    shaders.push_back(tdogl::Shader::shaderFromFile(resourcePath + "../resources/fragment-shader.txt", GL_FRAGMENT_SHADER));

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

    drawFlowField();

    glfwSwapBuffers(gWindow);
}

void Visualizer::drawFlowField()
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(gProgram->object());

    glBindVertexArray(vertexArrayID);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferID);

    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);
}

void Visualizer::mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods)
{
    Visualizer* visualizer = (Visualizer*) glfwGetWindowUserPointer(window);

    visualizer->mouseButtonCallback( window, button, action, mods );
}

void Visualizer::mouseMotionCallbackWrapper(GLFWwindow* window, double xpos, double ypos)
{
    Visualizer* visualizer = (Visualizer*) glfwGetWindowUserPointer(window);

    visualizer->mouseMotionCallback( window, xpos, ypos );
}

void Visualizer::keyboardCallbackWrapper(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Visualizer* visualizer = (Visualizer*) glfwGetWindowUserPointer(window);

    visualizer->keyboardCallback(window, key, scancode, action, mods);
}

void Visualizer::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if( action == GLFW_PRESS )
        isMouseButtonPressed = true;
    if( action == GLFW_RELEASE )
        isMouseButtonPressed = false;

    if( button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS )
        delelteGeo = true;
    else
        delelteGeo = false;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    float x = float(xpos);
    float y = float(ypos);

    // std::cout << "Clicked at ( " << x << ", " << y << " )" << std::endl;
    
    int xIdx = float(nx) * float(                  x ) / float( nx*pxPerVertex ) ;
    int yIdx = float(ny) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;

    if(isMouseButtonPressed)
        if( mods & GLFW_MOD_SHIFT )
            solver->setGeoFloodFill( xIdx, yIdx, delelteGeo?GEO_FLUID:GEO_SOLID );
        else
            solver->setGeo( xIdx, yIdx, delelteGeo? GEO_FLUID : GEO_SOLID);

    xIdxLast = xIdx;
    yIdxLast = yIdx;
}







void Visualizer::mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if( !isMouseButtonPressed ) return;

    float x = float(xpos);
    float y = float(ypos);

    // std::cout << "Motioned at ( " << x << ", " << y << " )" << std::endl;
    
    int xIdx = float(nx) * float(                  x ) / float( nx*pxPerVertex ) ;
    int yIdx = float(ny) * float( ny*pxPerVertex - y ) / float( ny*pxPerVertex ) ;

    // std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    if( xIdx <  0 || xIdx > 1000000 ) xIdx = 0;
    if( yIdx <  0 || yIdx > 1000000 ) yIdx = 0;
    if( xIdx >= nx - 2 )              xIdx = nx - 2;
    if( yIdx >= ny - 2 )              yIdx = ny - 2;

    // std::cout << "Motioned at ( " << xIdx << ", " << yIdx << " )" << std::endl;

    solver->setGeo( xIdxLast, yIdxLast, xIdx, yIdx, delelteGeo? GEO_FLUID : GEO_SOLID);

    geoModified = true;

    xIdxLast = xIdx;
    yIdxLast = yIdx;

    this->solver->scaleColorMap();
}

void Visualizer::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    
    
    const char* charFilePath = nullptr;
    std::string filePath;
    std::stringstream titleString;
    char * title = new char [250];
    const char* titleC=title;
    titleString<<"edu::flow "<<nx<<" x "<<ny<<" ";
    if( action == GLFW_RELEASE ) return;

    int R;
    switch (key)
    {
        case GLFW_KEY_H:
            std::cout << "================================================================================" << std::endl;
            std::cout << "          H e l p :" << std::endl;
            std::cout << "================================================================================" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Left Mouse:      Draw solid nodes" << std::endl;
            std::cout << "Right Mouse:     Delete solid nodes" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Press Shift with left or right mouse button to fill or erase whole areas" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "b:  toggle BGK and Central Moment methods" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "g:  Reset solid nodes" << std::endl;
            std::cout << "r:  Reset flow field" << std::endl;
            std::cout << "c:  Draw cylinder" << std::endl;
            std::cout << "a:  Draw Airfoil" << std::endl;
			std::cout << "m:  Draw Mesh" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "p:  Show pressure field" << std::endl;
            std::cout << "v:  Show velocity field" << std::endl;
            std::cout << "s:  Scale color map" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "n:  Show performance" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Up:     increase viscosity by factor 10" << std::endl;
            std::cout << "Down:   reduce viscosity by factor 10" << std::endl;
            std::cout << "Right:  increase velocity by 0.01 dx/dt" << std::endl;
            std::cout << "Left:   reduce velocity by 0.01 dx/dt" << std::endl;
			std::cout << "1:      turn velocity clockwise" << std::endl;
			std::cout << "2:      turn velocity anti-clockwise" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Page Up:    increase time steps per frame by one" << std::endl;
            std::cout << "Page Down:  decrease time steps per frame by one" << std::endl;
			std::cout << "0:	set 20 frames per second" << std::endl;
            std::cout << "j:	output flow data" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "================================================================================" << std::endl;
            break;

        case GLFW_KEY_B:
            if( solver->getLBModel() == 'b' ){
                solver->setLBModel('c');
                std::cout << "Using central moment method!" << std::endl;


            }
            else{
                solver->setLBModel('b');
                std::cout << "Using BGK!" << std::endl;
                std::cout << "Using central moment method!" << std::endl;

            }
            break;

        case GLFW_KEY_W:
            if( solver->getGeoMode() == 'b' ){
                solver->setGeoMode('w');
            }
            else{
                solver->setGeoMode('b');
            }
            break;

        case GLFW_KEY_V:
            std::cout << "Show Velocity!" << std::endl;
            postProcessingType = 'v';
            solver->scaleColorMap();
            break;

        case GLFW_KEY_P:
            std::cout << "Show Pressure!" << std::endl;
            postProcessingType = 'p';
            solver->scaleColorMap();
            break;

        case GLFW_KEY_R:
            std::cout << "Initialize Distributions!" << std::endl;
            solver->initializeDistributions();
            break;

        case GLFW_KEY_G:
            std::cout << "Initialize Distributions!" << std::endl;
            solver->initializeGeo();
            break;

        case GLFW_KEY_S:
            std::cout << "Scale ColorMap!" << std::endl;
            solver->scaleColorMap();
            break;

        case GLFW_KEY_N:
            std::cout << "Performance: " << std::endl;
            std::cout << "    " << nups << " NUPS" << std::endl;
            std::cout << "    " << fps  << " FPS"  << std::endl;
            break;

        case GLFW_KEY_UP:
            this->solver->setNu( 10.0f * this->solver->getNu() );
            std::cout << "Viscosity = " << this->solver->getNu() << std::endl;
            //std::cout << "Re = " << this->solver->getSpeed()*lref*(this->solver->getNu()) << std::endl;
            break;

        case GLFW_KEY_DOWN:
            this->solver->setNu( 0.1f * this->solver->getNu() );
            std::cout << "Viscosity = " << this->solver->getNu() << std::endl;
            //std::cout << "Re = " << this->solver->getSpeed()*lref*(this->solver->getNu()) << std::endl;
            break;

        case GLFW_KEY_RIGHT:
            this->solver->setSpeed( this->solver->getSpeed() + 0.001f );
            std::cout << "U = " << this->solver->getU() << " V = " << this->solver->getV() << std::endl;
            //std::cout << "Re = " << this->solver->getSpeed()*lref*(this->solver->getNu()) << std::endl;
            break;

        case GLFW_KEY_LEFT:
            this->solver->setSpeed( this->solver->getSpeed() - 0.001f );
			std::cout << "U = " << this->solver->getU() << " V = " << this->solver->getV() << std::endl;
            //std::cout << "Re = " << this->solver->getSpeed()*lref*(this->solver->getNu()) << std::endl;
            break;

        case GLFW_KEY_PAGE_UP:
            this->timeStepsPerFrame++;
            std::cout << "dt / frame = " << this->timeStepsPerFrame << std::endl;
            break;

        case GLFW_KEY_PAGE_DOWN:
            this->timeStepsPerFrame--;
            if( this->timeStepsPerFrame < 1 ) this->timeStepsPerFrame = 1;
            std::cout << "dt / frame = " << this->timeStepsPerFrame << std::endl;
            break;

		case GLFW_KEY_1:
			this->solver->setAlpha(this->solver->getAlpha() + 2.f*3.141592654f/360.f);
			this->solver->setSpeed(this->solver->getSpeed());
			std::cout << "alpha = " << this->solver->getAlpha() << std::endl;
			break;

		case GLFW_KEY_2:
			this->solver->setAlpha(this->solver->getAlpha() - 2.f*3.141592654f / 360.f);
			this->solver->setSpeed(this->solver->getSpeed());
			std::cout << "alpha = " << this->solver->getAlpha() << std::endl;
			break;

		case GLFW_KEY_0:
			this->timeStepsPerFrame = (uint)(fps*timeStepsPerFrame / 20.f + 1);
			std::cout << "dt / frame = " << this->timeStepsPerFrame << std::endl;
			break;

        case GLFW_KEY_C:
            R = this->ny/4;
            for( int x = -R; x <= R; x++ ){
                for( int y = -R; y <= R; y++ ){
                    if( sqrt( x * x + y * y ) > R ) continue;
                    solver->setGeo(2*R + x, 2*R + y,GEO_SOLID );
                }
            }
            break;

		case GLFW_KEY_M:
			for (int y = 1; y <= ny; y++) {
					if (y%11 ==0) solver->setGeo(2, y, GEO_SOLID);
					if (y % 13 == 0) solver->setGeo(5, y, GEO_SOLID);
			}
			break;

        case GLFW_KEY_K:
			for (int y = 1; y <= (2*ny)/3; y++) {
                solver->setGeo(1,y, GEO_SOLID);
                solver->setGeo(nx-3,y, GEO_SOLID);
                for (int x=1; x<=nx;x++){
                    solver->setGeo(x,2, GEO_SOLID);
                    if (y/2%67<45 && (y/2)% 71<67 && (x/2)%41<37 && (x/2)%47<31) solver->setGeo(x,y, GEO_SOLID);
                }

			}
			break;

        case GLFW_KEY_A:
            for( uint idx = 0; idx < nacaProfilePoints; idx++  ){

                float x1 = this->ny/8 + this->ny/2 * nacaProfile[(idx  )%nacaProfilePoints][0];
                float y1 = this->ny/2 + this->ny/2 * nacaProfile[(idx  )%nacaProfilePoints][1];
                float x2 = this->ny/8 + this->ny/2 * nacaProfile[(idx+1)%nacaProfilePoints][0];
                float y2 = this->ny/2 + this->ny/2 * nacaProfile[(idx+1)%nacaProfilePoints][1];

                solver->setGeo( x1, y1, x2, y2, GEO_SOLID);
            }

            solver->setGeoFloodFill( this->ny/8 + 10, this->ny/2, GEO_SOLID);

            break;

        case GLFW_KEY_D:
            for( uint idx = 0; idx < carGeometryPoints; idx++  ){

                float x1 = carGeometry[(idx  )%carGeometryPoints][0];
                float x2 = carGeometry[(idx+1)%carGeometryPoints][0];

                float y1 = this->ny/2 - carGeometry[(idx  )%carGeometryPoints][1];
                float y2 = this->ny/2 - carGeometry[(idx+1)%carGeometryPoints][1];

                solver->setGeo( x1, y1, x2, y2, GEO_SOLID);
            }

            solver->setGeoFloodFill( this->nx/2, this->ny/6, GEO_SOLID);

            R = 40;
            for( int x = -R; x <= R; x++ ){
                for( int y = -R; y <= R; y++ ){
                    if( sqrt( x * x + y * y ) > R ) continue;
                    solver->setGeo(180 + x, R + y,GEO_SOLID );
                }
            }

            R = 40;
            for( int x = -R; x <= R; x++ ){
                for( int y = -R; y <= R; y++ ){
                    if( sqrt( x * x + y * y ) > R ) continue;
                    solver->setGeo(500 + x, R + y,GEO_SOLID );
                }
            }

            break;


            case GLFW_KEY_Q:
            {
  
                std::string filePath = "geometry.bmp";
                command = std::make_unique<ReadSolidGeometryCommand>(filePath, solver);
                command->execute();

                if (currentCommand < static_cast<int>(history.size()) - 1) {
                    history.erase(history.begin() + currentCommand + 1, history.end());
                }
                history.push_back(std::move(command));
                currentCommand++;
  
              
            }
            break;

             case GLFW_KEY_L:
             
                command = std::make_unique<writeFlowFieldToVTKCommand>("flow_field_data.vtk", this->nx, this->ny,solver->getVelocityData() , solver->getPressureData());
                command->execute();
                if (currentCommand < static_cast<int>(history.size()) - 1) {
                    history.erase(history.begin() + currentCommand + 1, history.end());
                }
                history.push_back(std::move(command));
                currentCommand++;
            
       
        
            
            break;

            case GLFW_KEY_J:
             
                if (currentCommand >= 0 && currentCommand < static_cast<int>(history.size())) {
                history[currentCommand]->undo();
                currentCommand--;
            }
           
       
        
            
            break;

            case GLFW_KEY_T:

            {
                solver->setGeo(50, 270, 50, -10, GEO_SOLID);
                solver->setGeo(50, 330, 50, 610, GEO_SOLID);

                solver->setGeo(50, 270, 100, 290, GEO_SOLID);
                solver->setGeo(50, 330, 100, 310, GEO_SOLID);

                solver->setGeo(100, 290, 160, 290, GEO_SOLID);
                solver->setGeo(100, 310, 150, 310, GEO_SOLID);

                int L = 330;

                solver->setGeo(170, 300, L,   300, GEO_SOLID);
                solver->setGeo(170, 300, 230, 320, GEO_SOLID);
                solver->setGeo(230, 320, L,   320, GEO_SOLID);
                solver->setGeo(L  , 300, L,   320, GEO_SOLID);

                solver->setGeo(160, 290, 160, 250, GEO_SOLID);
                solver->setGeo(160, 250, L,   250, GEO_SOLID);

                // solver->setGeoFloodFill( 300, 310, GEO_SOLID);
            }

            break;

        default:
            break;
    }
                titleString<<((solver->getLBModel()=='b') ? "Collision: BGK": "Collision: CUMULANT");
                titleString<<" Re="<<solver->getRefLength()*solver->getSpeed()/solver->getNu();
                std::strcpy(title,titleString.str().c_str());
                glfwSetWindowTitle(gWindow,titleC);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint Visualizer::getVertexBufferID()
{
    return vertexBufferID;
}

