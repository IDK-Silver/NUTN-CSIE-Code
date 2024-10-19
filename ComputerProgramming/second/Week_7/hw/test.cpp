#include <iostream>
#include <utility>
#include <deque>
#include <cstring>
#define N 100
using namespace std;

int map[N][N];
typedef pair<int, int> p;
int n,m,dis=0,routes=1;
deque<p> path;
p pre[N][N];
int nxt[4][2] = {{-1,0},{0,1},{1,0},{0,-1}};

void bfs(int x, int y){
    path.push_back(make_pair(x,y));
    pre[x][y].first=x;
    pre[x][y].second=y;
    while (path.front().first!=n-1 || path.front().second!=m-1){
        if (routes==0) return;
        p from=path.front();
        path.pop_front();
        for (int i=0; i<4; i++){
            p to;
            to.first=from.first+nxt[i][0];
            to.second=from.second+nxt[i][1];
            if (to.first<0 || to.first>=n || to.second<0 || to.second>=m) continue;
            if (map[to.first][to.second] || pre[to.first][to.second].first != -1) continue;
            routes++;
            path.push_back(to);
            pre[to.first][to.second].first=from.first;
            pre[to.first][to.second].second=from.second;
        }
        routes--;
    }
}

void print(int x, int y){
    if(pre[x][y].first != x || pre[x][y].second != y){
        print(pre[x][y].first, pre[x][y].second);
        dis++;
    }
}

int main(){
    cin>>n>>m;
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            cin>>map[i][j];
    memset(pre,-1,sizeof(pre));
    bfs(0,0);
    print(n-1,m-1);
    if (routes==0)
        cout<<"No routes accessible.\n";
    else
        cout<<"Shortest distance: "<<dis<<endl;
    return 0;
}