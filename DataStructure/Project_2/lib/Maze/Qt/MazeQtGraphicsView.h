//
// Created by idk on 2023/10/30.
//

#ifndef MAZE_MAZE_QT_GRAPHICS_H
#define MAZE_MAZE_QT_GRAPHICS_H

#include <QGraphicsView>
#include <QGraphicsRectItem>
#include "MazeQtGraphicsPixmapItem.h"

class MazeQtGraphicsView : public QGraphicsView{
Q_OBJECT
public:
    explicit MazeQtGraphicsView(QWidget *parent = nullptr);

    ~MazeQtGraphicsView() override = default;

private:
    QGraphicsView* view{};

protected:
    void mousePressEvent(QMouseEvent *event) override;
};


#endif //MAZE_MAZE_QT_GRAPHICS_H
