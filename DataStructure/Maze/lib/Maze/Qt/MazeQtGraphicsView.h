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


    /* set get maze object */
    MazeObject get(size_t row, size_t column);
    void setMazeObject(size_t row, size_t column, MazeObject obj);

    /* set or get Painting mode */
    void setPaintingMode(bool flag);
    [[nodiscard]] bool isPaintingMode() const;

    /* set special flag */
    void setNextIsStartPoint();
    void setNextIsEndPoint();

private:
    // origin maze data
    std::shared_ptr<Maze> originMaze;

    // the maze start and end point
    std::pair<size_t, size_t> start_point, end_point;

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

    QPointF lastMousePos;

    bool isPainting = false;
    

Q_SIGNALS:
    void completedStartPoint(size_t row, size_t column);
    void completedEndPoint(size_t row, size_t column);
};


#endif //MAZE_MAZE_QT_GRAPHICS_H
