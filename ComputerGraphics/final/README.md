## Bowling is all you need
> The best game for fun after final exam

## How to get this game

if you want to play this game, you need to build yourself from source code

this is the build step (Note that the CMakeLists only supports MacOS, if you want to build on another OS, you will need to modify CMakeList)

1. clone this repo

   ```
   git clone https://github.com/IDK-Silver/NUTN-CSIE-Code.git
   ```

2. change work dir to this game folder

   ```
   cd NUTN-CSIE-Code/ComputerGraphics/final
   ```

3. create build folder and enter it

   ```
   mkdir build
   cd build
   ```

4. build game

   ```
   cmake ..
   cmake ---build . 
   ```

5. move shader to folder

   ```
   cp -r ../res/shader/* .
   ```

6. lunche game and enjoin it

   ```
   ./bowl-is-all-you-need
   ```

   

   
