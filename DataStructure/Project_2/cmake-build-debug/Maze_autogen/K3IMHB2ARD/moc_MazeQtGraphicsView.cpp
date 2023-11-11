/****************************************************************************
** Meta object code from reading C++ file 'MazeQtGraphicsView.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../lib/Maze/Qt/MazeQtGraphicsView.h"
#include <QtCore/qmetatype.h>

#if __has_include(<QtCore/qtmochelpers.h>)
#include <QtCore/qtmochelpers.h>
#else
QT_BEGIN_MOC_NAMESPACE
#endif


#include <memory>

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MazeQtGraphicsView.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS_t {};
static constexpr auto qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS = QtMocHelpers::stringData(
    "MazeQtGraphicsView",
    "completedStartPoint",
    "",
    "size_t",
    "row",
    "column",
    "completedEndPoint"
);
#else  // !QT_MOC_HAS_STRING_DATA
struct qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS_t {
    uint offsetsAndSizes[14];
    char stringdata0[19];
    char stringdata1[20];
    char stringdata2[1];
    char stringdata3[7];
    char stringdata4[4];
    char stringdata5[7];
    char stringdata6[18];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS_t qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS = {
    {
        QT_MOC_LITERAL(0, 18),  // "MazeQtGraphicsView"
        QT_MOC_LITERAL(19, 19),  // "completedStartPoint"
        QT_MOC_LITERAL(39, 0),  // ""
        QT_MOC_LITERAL(40, 6),  // "size_t"
        QT_MOC_LITERAL(47, 3),  // "row"
        QT_MOC_LITERAL(51, 6),  // "column"
        QT_MOC_LITERAL(58, 17)   // "completedEndPoint"
    },
    "MazeQtGraphicsView",
    "completedStartPoint",
    "",
    "size_t",
    "row",
    "column",
    "completedEndPoint"
};
#undef QT_MOC_LITERAL
#endif // !QT_MOC_HAS_STRING_DATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSMazeQtGraphicsViewENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    2,   26,    2, 0x06,    1 /* Public */,
       6,    2,   31,    2, 0x06,    4 /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 3,    4,    5,
    QMetaType::Void, 0x80000000 | 3, 0x80000000 | 3,    4,    5,

       0        // eod
};

Q_CONSTINIT const QMetaObject MazeQtGraphicsView::staticMetaObject = { {
    QMetaObject::SuperData::link<QGraphicsView::staticMetaObject>(),
    qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSMazeQtGraphicsViewENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS_t,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<MazeQtGraphicsView, std::true_type>,
        // method 'completedStartPoint'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<size_t, std::false_type>,
        QtPrivate::TypeAndForceComplete<size_t, std::false_type>,
        // method 'completedEndPoint'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<size_t, std::false_type>,
        QtPrivate::TypeAndForceComplete<size_t, std::false_type>
    >,
    nullptr
} };

void MazeQtGraphicsView::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MazeQtGraphicsView *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->completedStartPoint((*reinterpret_cast< std::add_pointer_t<size_t>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<size_t>>(_a[2]))); break;
        case 1: _t->completedEndPoint((*reinterpret_cast< std::add_pointer_t<size_t>>(_a[1])),(*reinterpret_cast< std::add_pointer_t<size_t>>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (MazeQtGraphicsView::*)(size_t , size_t );
            if (_t _q_method = &MazeQtGraphicsView::completedStartPoint; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (MazeQtGraphicsView::*)(size_t , size_t );
            if (_t _q_method = &MazeQtGraphicsView::completedEndPoint; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
    }
}

const QMetaObject *MazeQtGraphicsView::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MazeQtGraphicsView::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSMazeQtGraphicsViewENDCLASS.stringdata0))
        return static_cast<void*>(this);
    return QGraphicsView::qt_metacast(_clname);
}

int MazeQtGraphicsView::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void MazeQtGraphicsView::completedStartPoint(size_t _t1, size_t _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MazeQtGraphicsView::completedEndPoint(size_t _t1, size_t _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
