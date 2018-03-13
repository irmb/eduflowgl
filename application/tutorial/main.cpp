/*
 main

 Copyright 2012 Thomas Dalling - http://tomdalling.com/

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

//#include "platform.hpp"

// third-party libraries
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// standard C++ libraries
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>

// tdogl classes
#include "Program.h"

#include "cudaKernel.h"

// constants
const glm::vec2 SCREEN_SIZE(800, 600);

// globals
GLFWwindow* gWindow = NULL;
tdogl::Program* gProgram = NULL;
GLuint gVAO = 0;
GLuint gVBO = 0;
GLuint gEBO = 0;


float delta = 0.001f;
bool isMouseButtonPressed = false;

// loads the vertex shader and fragment shader, and links them to make the global gProgram
static void LoadShaders() {
    std::vector<tdogl::Shader> shaders;
#ifdef __APPLE__
    shaders.push_back(tdogl::Shader::shaderFromFile("../application/tutorial/resources/vertex-shader.txt", GL_VERTEX_SHADER));
    shaders.push_back(tdogl::Shader::shaderFromFile("../application/tutorial/resources/fragment-shader.txt", GL_FRAGMENT_SHADER));
#else
    shaders.push_back(tdogl::Shader::shaderFromFile("../../../application/tutorial/resources/vertex-shader.txt", GL_VERTEX_SHADER));
    shaders.push_back(tdogl::Shader::shaderFromFile("../../../application/tutorial/resources/fragment-shader.txt", GL_FRAGMENT_SHADER));
#endif
    gProgram = new tdogl::Program(shaders);
}


// loads a triangle into the VAO global
static void LoadTriangle() {
    // make and bind the VAO
    glGenVertexArrays(1, &gVAO);
    glBindVertexArray(gVAO);
    
    // make and bind the VBO
    glGenBuffers(1, &gVBO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    
    // Put the three triangle verticies into the VBO
    GLfloat vertexData[] = {
        //  X     Y     Z
         0.0f, 0.8f, 0.0f, 0.0, 1.0, 0.0,
        -0.8f,-0.8f, 0.0f, 1.0, 1.0, 0.0,
         0.8f,-0.8f, 0.0f, 0.0, 1.0, 1.0
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);
    
    // connect the xyz to the "vert" attribute of the vertex shader
    glEnableVertexAttribArray(gProgram->attrib("vert"));
    glVertexAttribPointer(gProgram->attrib("vert"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, NULL);

    glEnableVertexAttribArray(gProgram->attrib("color"));
    glVertexAttribPointer(gProgram->attrib("color"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (char*)(sizeof(float) * 3));
    
    // unbind the VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

static void loadElement(){

    glGenBuffers(1, &gEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);

    GLuint element [] = {
        0, 1, 2
    };

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(element), element, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// draws a single frame
static void Render() {
    // clear everything
    glClearColor(0, 0, 0, 1); // black
    glClear(GL_COLOR_BUFFER_BIT);
    
    // bind the program (the shaders)
    glUseProgram(gProgram->object());

    // draw the VAO
    //glDrawArrays(GL_TRIANGLES, 0, 3);
    
    glBindVertexArray(gVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);

    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBindVertexArray(0);
    
    // unbind the program
    glUseProgram(0);
    
    // swap the display buffers (displays what was just drawn)
    glfwSwapBuffers(gWindow);
}

void OnError(int errorCode, const char* msg) {
    throw std::runtime_error(msg);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if( action == GLFW_PRESS ){
        if (key == GLFW_KEY_UP)
            delta += 0.001f;
        if (key == GLFW_KEY_DOWN)
            delta -= 0.001f;
    }
}

void mouse_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if( isMouseButtonPressed )
        std::cout << "( " << xpos << " , " << ypos << " )" << std::endl;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if( button == GLFW_MOUSE_BUTTON_LEFT ){
        
        if( action == GLFW_PRESS )
            isMouseButtonPressed = true;
        if( action == GLFW_RELEASE )
            isMouseButtonPressed = false;
    }
}

// the program starts here
void AppMain() {
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
    gWindow = glfwCreateWindow((int)SCREEN_SIZE.x, (int)SCREEN_SIZE.y, "OpenGL Tutorial", NULL, NULL);
    if(!gWindow)
        throw std::runtime_error("glfwCreateWindow failed. Can your hardware handle OpenGL 3.2?");

    // GLFW settings
    glfwMakeContextCurrent(gWindow);
    
    glfwSetKeyCallback(gWindow, key_callback);

    glfwSetCursorPosCallback(gWindow, mouse_position_callback);

    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);

    // initialise GLEW
    glewExperimental = GL_TRUE; //stops glew crashing on OSX :-/
    if(glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");

    // print out some info about the graphics drivers
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

    // make sure OpenGL version 3.2 API is available
    if(!GLEW_VERSION_3_2)
        throw std::runtime_error("OpenGL 3.2 API is not available.");

    // load vertex and fragment shaders into opengl
    LoadShaders();

    // create buffer and fill it with the points of the triangle
    LoadTriangle();

    loadElement();

    connectVertexBuffer( gVBO );

    // run while the window is open
    while(!glfwWindowShouldClose(gWindow)){
        // process pending events
        glfwPollEvents();
        
        changeTriangle( delta );

        // draw one frame
        Render();
    }

    // clean up and exit
    glfwTerminate();
}


int main(int argc, char *argv[]) {
    try {
        AppMain();
    } catch (const std::exception& e){
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}





