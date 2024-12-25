#include "main_menu.h"

Menu* Menu::currentMenu = nullptr;

Menu::Menu() {
    currentMenu = this;
}

void Menu::addEntry(const std::string& name, std::function<void(int)> action) {
    items.push_back({name, action});
}

void Menu::addSubMenu(const std::string& name, Menu& subMenu) {
    subMenus.push_back(&subMenu);
}

void Menu::create() {
    for (auto& subMenu : subMenus) {
        subMenu->create();
    }

    int menuId = glutCreateMenu(processMenu);
    for (size_t i = 0; i < items.size(); ++i) {
        glutAddMenuEntry(items[i].name.c_str(), i);
    }
    for (size_t i = 0; i < subMenus.size(); ++i) {
        glutAddSubMenu(subMenus[i]->items[0].name.c_str(), glutGetMenu());
    }
}

void Menu::attach(int button) {
    glutAttachMenu(button);
}

void Menu::processMenu(int id) {
    if (id >= 0 && id < currentMenu->items.size()) {
        currentMenu->items[id].action(id);
    }
}