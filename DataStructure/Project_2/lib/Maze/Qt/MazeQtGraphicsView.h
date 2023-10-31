//
// Created by idk on 2023/10/30.
//

#ifndef MAZE_MAZE_QT_GRAPHICS_H
#define MAZE_MAZE_QT_GRAPHICS_H

#include <memory>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include "MazeQtGraphicsPixmapItem.h"

class MazeQtGraphicsView : public QGraphicsView{
Q_OBJECT
public:
    explicit MazeQtGraphicsView(QWidget *parent = nullptr);
    ~MazeQtGraphicsView() override;

    /* the method to create graphics maze */
    void createMaze(std::shared_ptr<Maze> obj);


    /* get maze object */
    MazeObject get(size_t row, size_t column);

    /* set or get Painting mode */
    void setPaintingMode(bool flag);
    bool isPaintingMode() const;

    /* set special flag */
    void setNextIsStartPoint();
    void setNextIsEndPoint();

private:
    std::shared_ptr<Maze> originMaze;

protected:
    /* the Qt slot */
    void mousePressEvent(QMouseEvent *event) override;

    /* the Qt graphics scene */
    QGraphicsScene *scene;

    /* the graphics maze */
    std::shared_ptr<QVector<QVector<MazeQtGraphicsPixmapItem*>>> maze;

    /* painting mode */
    bool _isPaintingMode = false;

    /* special flag */
    bool _nextIsSetStartPoint = false;
    bool _nextIsSetEndPoint = false;
Q_SIGNALS:
    void completedStartPoint(size_t row, size_t column);
    void completedEndPoint(size_t row, size_t column);
};


#endif //MAZE_MAZE_QT_GRAPHICS_H
