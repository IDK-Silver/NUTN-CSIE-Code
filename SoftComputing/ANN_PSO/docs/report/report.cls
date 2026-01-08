% report.cls - 簡潔報告文件類別
% 適用於 macOS + XeLaTeX
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{report}[2024/01/01 Custom Report Class]

% 基於 article 類別
\LoadClass[12pt, a4paper]{article}

% 必要套件
\RequirePackage{fontspec}
\RequirePackage{xeCJK}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{float}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{listings}

% 頁面設定
\geometry{
    top=2.5cm,
    bottom=2.5cm,
    left=2.5cm,
    right=2.5cm,
    headheight=14pt
}

% macOS 中文字體設定
\setCJKmainfont{PingFang TC}[
    UprightFont=*-Regular,
    BoldFont=*-Semibold
]
\setCJKsansfont{PingFang TC}[
    UprightFont=*-Regular,
    BoldFont=*-Semibold
]
\setCJKmonofont{PingFang TC}

% 英文字體設定
\setmainfont{Helvetica Neue}
\setsansfont{Helvetica Neue}
\setmonofont{Menlo}[Scale=0.9]

% 顏色定義
\definecolor{primary}{RGB}{41, 128, 185}
\definecolor{secondary}{RGB}{52, 73, 94}
\definecolor{accent}{RGB}{230, 126, 34}
\definecolor{codebg}{RGB}{248, 248, 248}
\definecolor{codeframe}{RGB}{220, 220, 220}

% 超連結設定
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    urlcolor=primary,
    citecolor=primary
}

% 標題格式
\titleformat{\section}
    {\Large\bfseries\color{secondary}}
    {\thesection}{1em}{}
\titleformat{\subsection}
    {\large\bfseries\color{secondary}}
    {\thesubsection}{1em}{}
\titleformat{\subsubsection}
    {\normalsize\bfseries\color{secondary}}
    {\thesubsubsection}{1em}{}

% 標題間距
\titlespacing*{\section}{0pt}{2ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.5ex plus 1ex minus .2ex}{1ex plus .2ex}

% 頁首頁尾
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{secondary}軟性計算 Homework \#02}
\fancyhead[R]{\small\color{secondary}S11159005 黃毓峰}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% 圖表標題格式
\captionsetup{
    font=small,
    labelfont=bf,
    labelsep=period,
    skip=8pt
}

% 程式碼格式
\lstset{
    basicstyle=\small\ttfamily,
    backgroundcolor=\color{codebg},
    frame=single,
    rulecolor=\color{codeframe},
    breaklines=true,
    breakatwhitespace=true,
    showstringspaces=false,
    numbers=left,
    numberstyle=\tiny\color{gray},
    numbersep=8pt,
    tabsize=4
}

% 列表格式
\setlist[itemize]{itemsep=0.5ex, topsep=0.5ex, leftmargin=*, labelindent=0pt}
\setlist[enumerate]{itemsep=0.5ex, topsep=0.5ex, leftmargin=*, labelindent=0pt}

% 封面指令
\newcommand{\makecover}[4]{%
    \begin{titlepage}
        \centering
        \vspace*{3cm}
        {\LARGE\bfseries #1\par}
        \vspace{1.5cm}
        {\Huge\bfseries #2\par}
        \vspace{2cm}
        {\Large #3\par}
        \vspace{1cm}
        {\Large #4\par}
        \vfill
        {\large \today\par}
    \end{titlepage}
}

% 表格美化
\renewcommand{\arraystretch}{1.3}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\endinput
