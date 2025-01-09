#ifndef MENU_H
#define MENU_H

#include <GL/glut.h>
#include <functional>
#include <vector>
#include <string>

struct MenuItem {
    std::string name;
    std::function<void(int)> action;
};

class Menu {
public:
    Menu();
    void addEntry(const std::string& name, std::function<void(int)> action);
    void addSubMenu(const std::string& name, Menu& subMenu);
    void create();
    void attach(int button);

private:
    std::vector<MenuItem> items;
    std::vector<Menu*> subMenus;
    static void processMenu(int id);
    static Menu* currentMenu;
};

#endif // MENU_H