//
// Created by idk on 2023/10/25.
//

#ifndef MAZE_MAINWINDOW_H
#define MAZE_MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#include <QMainWindow>
#include <QGraphicsItem>
#include <lib/Maze/Maze.h>
#include <lib/Maze/Qt/MazeQtGraphicsView.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

    ~MainWindow() override;


private:
    Ui::MainWindow *ui{};

    /* the origin maze data */
    std::shared_ptr<Maze> maze;

    /* Qt slot connect initial */
    void connectSetUp();

private slots:
    void createMaze();

};

#endif //MAZE_MAINWINDOW_H
