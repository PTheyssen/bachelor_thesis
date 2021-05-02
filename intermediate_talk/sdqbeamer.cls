%% Modified LaTeX-Beamer template, stolen from the SDQ group and
%% adapted by Philipp Becker 

%% After we have a group logo we need to uncomment some things here
%% to restore the functionality, just search for "grouplogo"

%% Original Statement:
%%-------------------------
%% LaTeX-Beamer template for KIT design
%% by Erik Burger, Christian Hammer
%% title picture by Klaus Krogmann
%%
%% version 2.5
%%
%% mostly compatible to KIT corporate design v2.0
%% http://intranet.kit.edu/gestaltungsrichtlinien.php
%%
%% Problems, bugs and comments to
%% burger@kit.edu

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sdqbeamer}[2020-07-22 v2.5 SDQ Beamer class]

\newif\ifsectionnavigation
\newif\ifnavbarinfoot
\newif\ifnavbarinline
\newif\ifnavbarside
\newif\ifwidescreen
\newif\ifgerman

\def\kitslogan#1{\def\@kitslogan{#1}}
\def\kitlogo#1{\def\@kitlogo{#1}}
\def\groupname#1{\def\@groupname{#1}}

\DeclareOption{de}{
  \kitslogan{KIT -- Die Forschungsuniversität in der Helmholtz-Gemeinschaft}
  \groupname{Autonome Lernende Roboter, Institut für Anthropomatik und Robotik, KIT-Fakultät für Informatik}
  \kitlogo{kitlogo_de_rgb}
  \germantrue
  \PassOptionsToPackage{autostyle}{csquotes}
  }
\DeclareOption{en}{
  \kitslogan{KIT -- The Research University in the Helmholtz Association}
  \groupname{Autonomous Learning Robots, Institute for Anthropomatics and Robotics
, KIT Department of Informatics}
  \kitlogo{kitlogo_en_rgb}
  \germanfalse
  }
\DeclareOption{4:3}{
  \widescreenfalse
}
\DeclareOption{16:9}{
  \widescreentrue
}
\DeclareOption{navbarside}{
  \sectionnavigationtrue
  \navbarsidetrue
  \navbarinlinefalse
  \navbarinfootfalse
}
\DeclareOption{navbarinline}{
  \sectionnavigationtrue
  \navbarsidefalse
  \navbarinlinetrue
  \navbarinfootfalse
}
\DeclareOption{navbarinfooter}{
  \sectionnavigationtrue
  \navbarsidefalse
  \navbarinlinefalse
  \navbarinfoottrue
}
\DeclareOption{navbaroff}{
  \sectionnavigationfalse
}

\ExecuteOptions{en,16:9,navbarinline}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}

\ProcessOptions\relax

\LoadClass[18pt,utf8,professionalfonts,compress]{beamer}

% Aspect ratio
\RequirePackage{geometry}
\ifwidescreen
  \geometry{papersize={17.07cm,9.6cm}}
  \else
  \geometry{papersize={12.8cm,9.6cm}}
\fi

% Load babel only if needed (German)
\ifgerman
  \RequirePackage[ngerman]{babel}
\fi
\RequirePackage{csquotes}
\RequirePackage{hyperref}
\RequirePackage[absolute,overlay]{textpos}

%% --------------
%% | Typography |
%% --------------
%% Philipp: Cleaned up here a bit because the equations where 
%% just ugly... 
\RequirePackage[utf8]{inputenc}

\usefonttheme[onlymath]{serif}
\RequirePackage[scaled=.92]{helvet}
\RequirePackage[scaled=.78]{beramono}


\setbeamerfont{title}{series=\bfseries,size=\Large}
\setbeamerfont{frametitle}{series=\bfseries}

%% ----------------
%% | Slide Layout |
%% ----------------

\setbeamersize{text margin left=5.5mm,text margin right=5.5mm} 

% no navigations ymbols
\setbeamertemplate{navigation symbols}{}
\setbeamercovered{invisible}
\useinnertheme{rounded}


%frametitle
\setbeamertemplate{frametitle}{
    \begin{textblock*}{\dimexpr\paperwidth-30mm}[0,.5](5.5mm,9mm)
      \usebeamerfont{frametitle}\insertframetitle
    \end{textblock*}
    \begin{textblock*}{16mm}(\dimexpr\paperwidth-21.5mm\relax,5mm)
        \includegraphics[width=16mm]{logos/\@kitlogo}
    \end{textblock*}%
}

% top margin
\setbeamertemplate{headline}{
  \begin{beamercolorbox}[wd=\paperwidth,ht=12mm,dp=0pt]{}%
  \end{beamercolorbox}
}

%%
%% Footer
%%

\newlength{\kitbottom}
  %
  % if navbar in footer: slim footline
  %
\ifnavbarinfoot
  \setbeamerfont{footer}{size=\fontsize{6pt}{7.2pt}\selectfont}
  \setlength{\kitbottom}{4mm}
\else
  \setbeamerfont{footer}{size=\scriptsize}
  \setlength{\kitbottom}{9mm}
\fi
\setbeamerfont{page number in head/foot}{series=\bfseries}

\newlength{\kitbottomdepth}
\newlength{\kitbottomheight}

\setbeamertemplate{footline}{
  % need "%" at the line ends lest whitespaces are introduced
  \usebeamerfont{footer}%
  \ifsectionnavigation 
    \ifnavbarinline
      \begin{beamercolorbox}[wd=\paperwidth, leftskip=2.5mm, rightskip=2.5mm]{}
        \insertnavigation{\dimexpr\paperwidth-5mm\relax}
        \vspace{1mm}
      \end{beamercolorbox}%
    \fi
    \ifnavbarinfoot
      \begin{beamercolorbox}[wd=\paperwidth, leftskip=2.5mm, rightskip=19mm]{}
        \insertnavigation{\dimexpr\paperwidth-21.5mm\relax}
      \end{beamercolorbox}%
    \fi
  \fi%
  \leavevmode%
  \setlength{\kitbottomheight}{\dimexpr.5\kitbottom+.5em\relax}%
  \setlength{\kitbottomdepth}{\dimexpr.5\kitbottom-.5em\relax}%
  \begin{beamercolorbox}[wd=13mm, ht=\kitbottomheight, dp=\kitbottomdepth, leftskip=5.5mm]{}
      \usebeamerfont{page number in head/foot}%
      \strut\insertframenumber{}/\inserttotalframenumber%
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=20mm, ht=\kitbottomheight, dp=\kitbottomdepth]{}
      \usebeamerfont{date in head/foot}%
      \strut\insertshortdate%
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=\dimexpr\paperwidth-47mm\relax, ht=\kitbottomheight, dp=\kitbottomdepth]{}
      \hspace{0.5 cm} \insertshortauthor%: \insertshorttitle
  \end{beamercolorbox}%
 
  %\begin{beamercolorbox}[wd=14mm, ht=\kitbottomheight, dp=\kitbottomdepth, rightskip=5.5mm]{}
  %  \ifnavbarinfoot
  %    \raisebox{1.5mm}{\includegraphics[width=8.5mm]{logos/\@grouplogo}}
  %  \else
  %    \raisebox{-1mm}{\includegraphics[width=8.5mm]{logos/\@grouplogo}}
%    \fi
 % \end{beamercolorbox}%
}

%%
%% Sidebar
%%

\ifnavbarside
\useoutertheme[height=0cm,width=4cm,left]{sidebar}

\setbeamerfont{title in sidebar}{family=\sffamily,series=\mdseries,size={\fontsize{10pt}{11pt}}}
\setbeamerfont{section in sidebar}{family=\sffamily,series=\mdseries,size={\fontsize{9pt}{9.9pt}}}
\setbeamerfont{subsection in sidebar}{family=\sffamily,series=\mdseries,size={\fontsize{8pt}{8.8pt}}}

\setbeamertemplate{sidebar \beamer@sidebarside}
  {\vskip1.5cm%
   \hskip4.5mm%
   \advance\beamer@sidebarwidth by -5mm%
   \insertverticalnavigation{\beamer@sidebarwidth}%
  }%
\fi

%% ----------
%% | Colors |
%% ----------
\RequirePackage{templates/beamercolorthemekit}

\setbeamercolor*{normal text}{fg=black}
\setbeamercolor*{alerted text}{fg=kit-red100}
\setbeamercolor*{example text}{fg=black}
\setbeamercolor*{structure}{fg=black}


\setbeamercolor*{palette primary}{fg=black,bg=black!15}
\setbeamercolor*{palette secondary}{fg=black,bg=black!15}
\setbeamercolor*{palette tertiary}{fg=black,bg=black!15}
\setbeamercolor*{palette quaternary}{fg=black,bg=black!15}

\setbeamercolor*{palette sidebar primary}{fg=black!75}
\setbeamercolor*{palette sidebar secondary}{fg=black!75}
\setbeamercolor*{palette sidebar tertiary}{fg=black!75}
\setbeamercolor*{palette sidebar quaternary}{fg=black!75}

\setbeamercolor*{item projected}{fg=white,bg=kit-green100}

\setbeamercolor*{block title}{fg=white,bg=kit-green100}
\setbeamercolor*{block title alerted}{use=alerted text,fg=white,bg=alerted text.fg!75!black}
\setbeamercolor*{block title example}{fg=white,bg=kit-blue100}

\setbeamercolor*{block body}{fg=black,bg=kit-green15}
\setbeamercolor*{block body alerted}{parent=normal text,use=block title alerted,bg=block title alerted.bg!10!bg}
\setbeamercolor*{block body example}{fg=black,bg=kit-blue15}

\setbeamercolor*{separation line}{}
\setbeamercolor*{fine separation line}{}

\setbeamercolor*{background canvas}{bg=white}

%background for pages
\usebackgroundtemplate{
  \kitpageframevariable
}

\newcommand{\kitpageframevariable}{
  \begin{pgfpicture}{0mm}{0mm}{\paperwidth}{\paperheight}
  \kitpageframe{\ifbeamer@plainframe1.5mm\else9mm\fi}
  \end{pgfpicture}%
}

\newcommand{\kitpageframe}[1]{
    % whole page
    \pgfpathrectangle{\pgfpointorigin}{\pgfpoint{\paperwidth}{\paperheight}}
    % semirounded cut-out
    \pgfsetcornersarced{\pgfpoint{3mm}{3mm}}
    \pgfpathmoveto{\pgfpoint{\paperwidth-1.5mm}{#1}}
    \pgfpathlineto{\pgfpoint{1.5mm}{#1}}
    \pgfsetcornersarced{\pgfpointorigin}
    \pgfpathlineto{\pgfpoint{1.5mm}{\paperheight-1.5mm}}
    \pgfsetcornersarced{\pgfpoint{3mm}{3mm}}
    \pgfpathlineto{\pgfpoint{\paperwidth-1.5mm}{\paperheight-1.5mm}}
    \pgfpathlineto{\pgfpoint{\paperwidth-1.5mm}{#1}}
    \pgfsetcornersarced{\pgfpointorigin}
    \pgfpathclose

    \color{black!15}
    \pgfusepath{fill}
}


%% --------------
%% | Title Page |
%% --------------

%title image and logo
\def\titleimage#1{\def\@titleimage{#1}}
%\def\grouplogo#1{\def\@grouplogo{#1}}

\titleimage{bauplan_blau_wide}
%% ------------------------------------------
%% Add Our Logo Here once we have one
%% ------------------------------------------
%\grouplogo{}

% extra command for title frame to put the \titleimage behind the frame
\newcommand{\KITtitleframe}{
  \usebackgroundtemplate{
    \begin{pgfpicture}{0mm}{0mm}{\paperwidth}{\paperheight}
      % title image
      \pgftext[at=\pgfpoint{1mm}{7.5mm},left,bottom]{%
        \includegraphics[height=40mm]{logos/\@titleimage}
      }
      % dark grey frame for institute name
	    \color{black!50}
	    \pgfpathrectangle{\pgfpoint{1mm}{48mm}}{\pgfpoint{\paperwidth-2mm}{6mm}}
	    \pgfusepath{fill}
      % light grey frame
      \kitpageframe{9mm}
    \end{pgfpicture}%
  }
  \begin{frame}[plain]
  \titlepage
  \end{frame}
  % normal background
  \usebackgroundtemplate{
    \kitpageframevariable
  }
}

\setbeamertemplate{title page}{

  %middle block with name of institute
  \begin{textblock*}{\paperwidth}[0,.5](5.5mm,45mm)
    % tTODO: groupname
  \textcolor{white}{\fontsize{5pt}{5pt}\selectfont\uppercase\expandafter{\@groupname}}
  \end{textblock*}
  
  %elements on title page
  
  %KIT logo
  \begin{textblock*}{24mm}(5.5mm,5mm)
  \includegraphics[width=23mm]{logos/\@kitlogo} 
  \end{textblock*}
  
  %group logo
  \begin{textblock*}{16mm}(\dimexpr\paperwidth-21.5mm\relax,5mm)
  %\includegraphics[width=16mm]{logos/\@grouplogo} 
  \end{textblock*}
  
  %title
  \begin{textblock*}{\dimexpr\paperwidth-8mm\relax}[0,.5](5.5mm,25mm)
  %\begin{beamercolorbox}[wd=118mm,ht=14mm,dp=3mm,left]{white}%
  \usebeamerfont*{title}\inserttitle
  %\end{beamercolorbox}
  \end{textblock*}
  
  %subtitle
  \begin{textblock*}{\dimexpr\paperwidth-8mm\relax}(5.5mm,33mm)
  \scriptsize \insertsubtitle
  \end{textblock*}
  
  %author
  \begin{textblock*}{\dimexpr\paperwidth-8mm\relax}(5.5mm,37mm)
  \scriptsize\insertauthor~\textbar~\insertshortdate
  \end{textblock*}
  
  % KIT slogan
  \begin{textblock*}{80mm}[0,.5](5.5mm,91.5mm)
      \fontsize{6pt}{6pt}\selectfont\@kitslogan
  \end{textblock*}
  
  \begin{textblock*}{20mm}[0,.5](\dimexpr\paperwidth-25.5mm\relax,91.5mm)
      \fontsize{12pt}{12pt}\selectfont\bfseries\raggedleft%
      \textcolor{white}{\href{https://www.kit.edu}{www.kit.edu}}
  \end{textblock*}
  
}
%% ----------------
%% | /Title Slide |
%% ----------------

%% -----------------------------------------
%% | KIT-style semi-rounded itemize points |
%% -----------------------------------------

\setbeamertemplate{itemize items}{
  \raisebox{.25ex}{
    \begin{pgfpicture}{0mm}{0mm}{1ex}{1ex}
    {\pgfsetcornersarced{\pgfpoint{.3ex}{.3ex}} 
    \pgfpathmoveto{\pgfpoint{0cm}{1ex}} 
    \pgfpathlineto{\pgfpoint{1ex}{1ex}}
    \pgfpathlineto{\pgfpoint{1ex}{0cm}}}
    {\pgfsetcornersarced{\pgfpoint{.3ex}{.3ex}}
    \pgfpathmoveto{\pgfpoint{1ex}{0cm}}
    \pgfpathlineto{\pgfpointorigin}
    \pgfpathlineto{\pgfpoint{0cm}{1ex}}}
    \color{kit-green100}
    \pgfusepath{fill}
    \end{pgfpicture}%
  }
}

%% ---------------------------------------
%% | KIT-style semi-rounded beamer boxes |
%% ---------------------------------------

\renewcommand\beamerboxesrounded[2][]{%
  \global\let\beamer@firstlineitemizeunskip=\relax%
  \vbox\bgroup%
  \setkeys{beamerboxes}{upper=block title,lower=block body,width=\textwidth,shadow=false}%
  \setkeys{beamerboxes}{#1}%
  {%
    \usebeamercolor{\bmb@lower}%
    \globalcolorstrue%
    \colorlet{lower.bg}{bg}%
  }%
  {%
    \usebeamercolor{\bmb@upper}%
    \globalcolorstrue%
    \colorlet{upper.bg}{bg}%
  }%
  %
  % Typeset head
  %
  \vskip4bp
  \setbox\bmb@box=\hbox{%
    \begin{minipage}[b]{\bmb@width}%
      \usebeamercolor[fg]{\bmb@upper}%
      #2%
    \end{minipage}}%
  \ifdim\wd\bmb@box=0pt%
    \setbox\bmb@box=\hbox{}%
    \ht\bmb@box=1.5pt%
    \bmb@prevheight=-4.5pt%
  \else%
    \wd\bmb@box=\bmb@width%
    \bmb@temp=\dp\bmb@box%
    \ifdim\bmb@temp<1.5pt%
      \bmb@temp=1.5pt%
    \fi%
    \setbox\bmb@box=\hbox{\raise\bmb@temp\hbox{\box\bmb@box}}%
    \dp\bmb@box=0pt%
    \bmb@prevheight=\ht\bmb@box%
  \fi%
  \bmb@temp=\bmb@width%
  \bmb@dima=\bmb@temp\advance\bmb@dima by2.2bp%
  \bmb@dimb=\bmb@temp\advance\bmb@dimb by4bp%
  \hbox{%
    \begin{pgfpicture}{0bp}{+-\ht\bmb@box}{0bp}{+-\ht\bmb@box}
      \ifdim\wd\bmb@box=0pt%
        \color{lower.bg}%
      \else%        
        \color{upper.bg}%
      \fi%  
      \pgfpathqmoveto{-4bp}{-1bp}
      % Adaption for "KIT-Design"
      \pgfpathlineto{\pgfpoint{-4bp}{3bp}}
      %\pgfpathqcurveto{-4bp}{1.2bp}{-2.2bp}{3bp}{0bp}{3bp}
      \pgfpathlineto{\pgfpoint{\bmb@temp}{3bp}}
      \pgfpathcurveto%
      {\pgfpoint{\bmb@dima}{3bp}}%
      {\pgfpoint{\bmb@dimb}{1.2bp}}%
      {\pgfpoint{\bmb@dimb}{-1bp}}%
      \bmb@dima=-\ht\bmb@box%
      \advance\bmb@dima by-2pt%
      \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
      \pgfpathlineto{\pgfpoint{-4bp}{\bmb@dima}}
      \pgfusepath{fill}
    \end{pgfpicture}%
    \copy\bmb@box%
  }%
  \nointerlineskip%
  \vskip-1pt%
  \ifdim\wd\bmb@box=0pt%
  \else%
  \hbox{%
    \begin{pgfpicture}{0pt}{0pt}{\bmb@width}{6pt}
      \bmb@dima=\bmb@width%
      \advance\bmb@dima by8bp%
      \pgfpathrectangle{\pgfpoint{-4bp}{-1bp}}{\pgfpoint{\bmb@dima}{8bp}}
      \pgfusepath{clip}
      {\pgftransformshift{\pgfpoint{-4bp}{0bp}}\pgftext[left,base]{\pgfuseshading{bmb@transition}}}%
    \end{pgfpicture}%
  }%
  \nointerlineskip%
  \vskip-0.5pt%
  \fi%
  \ifbmb@shadow%
    \setbox\bmb@boxshadow=\hbox{\pgfuseshading{bmb@shadow}}%
    \setbox\bmb@boxshadowball=\hbox{\pgfuseshading{bmb@shadowball}}%
    \setbox\bmb@boxshadowballlarge=\hbox{\pgfuseshading{bmb@shadowballlarge}}%
  \fi%
  \setbox\bmb@colorbox=\hbox{{\pgfpicturetrue\pgfsetcolor{lower.bg}}}%
  \setbox\bmb@box=\hbox\bgroup\begin{minipage}[b]{\bmb@width}%
    \vskip2pt%
    \usebeamercolor[fg]{\bmb@lower}%
    \colorlet{beamerstructure}{upper.bg}%
    \colorlet{structure}{upper.bg}%
    %\color{.}%
  }

\def\endbeamerboxesrounded{%
  \end{minipage}\egroup%
  \wd\bmb@box=\bmb@width%
  \bmb@temp=\dp\bmb@box%
  \advance\bmb@temp by.5pt%
  \setbox\bmb@box=\hbox{\raise\bmb@temp\hbox{\box\bmb@box}}%
  \dp\bmb@box=0pt%
  \bmb@temp=\wd\bmb@box%
  \bmb@dima=\bmb@temp\advance\bmb@dima by2.2bp%
  \bmb@dimb=\bmb@temp\advance\bmb@dimb by4bp%
  \hbox{%
    \begin{pgfpicture}{0bp}{0bp}{0bp}{0bp}
      \ifbmb@shadow%
      {\pgftransformshift{\pgfpoint{4bp}{-3bp}}\pgftext{\copy\bmb@boxshadowball}}
      \begin{pgfscope}
        {%
        \advance\bmb@temp by-1bp%
        \pgfpathrectangle{\pgfpoint{\bmb@temp}{-7bp}}{\pgfpoint{9bp}{9bp}}%
        \pgfusepath{clip}
        }%
        {\pgftransformshift{\pgfpoint{\bmb@temp}{1bp}}\pgftext{\box\bmb@boxshadowballlarge}}
      \end{pgfscope}
      \begin{pgfscope}
        \advance\bmb@temp by-4bp%
        \pgfpathrectangle{\pgfpoint{4bp}{-7bp}}{\pgfpoint{\bmb@temp}{5bp}}
        \pgfusepath{clip}
        {\pgftransformshift{\pgfpoint{4bp}{-7bp}}\pgftext[left,base]{\copy\bmb@boxshadow}}%
      \end{pgfscope}
      \begin{pgfscope}
        \advance\bmb@temp by 4bp%
        \bmb@dima=\ht\bmb@box%
        \advance\bmb@dima by\bmb@prevheight%
        \advance\bmb@dima by 4bp%
        \pgfpathrectangle{\pgfpoint{\bmb@temp}{1bp}}{\pgfpoint{4bp}{\bmb@dima}}
        \pgfusepath{clip}
        \advance\bmb@dima by-4bp%
        {\pgftransformshift{\pgfpoint{\bmb@temp}{\bmb@dima}}\pgftext{\box\bmb@boxshadowball}}
        \advance\bmb@dima by-1bp%
        \pgfpathrectangle{\pgfpoint{\bmb@temp}{1bp}}{\pgfpoint{4bp}{\bmb@dima}}
        \pgfusepath{clip}
        \advance\bmb@temp by4bp%
        {\pgftransformshift{\pgfpoint{\bmb@temp}{-3bp}}%
          \pgftransformrotate{90}%
          \pgftext[left,base]{\box\bmb@boxshadow}}%
      \end{pgfscope}
      \fi%
      \unhbox\bmb@colorbox%
      \pgfpathqmoveto{-4bp}{1bp}
      \pgfpathqcurveto{-4bp}{-1.2bp}{-2.2bp}{-3bp}{0bp}{-3bp}
      %KIT-Design
      \pgfpathlineto{\pgfpoint{\the\bmb@dimb}{-3bp}}
      %\pgfpathcurveto%
      %{\pgfpoint{\the\bmb@dima}{-3bp}}%
      %{\pgfpoint{\the\bmb@dimb}{-1.2bp}}%
      %{\pgfpoint{\the\bmb@dimb}{1bp}}%
      {
      \bmb@dima=\ht\bmb@box%
      \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
      \pgfpathlineto{\pgfpoint{-4bp}{\bmb@dima}}
      \pgfusepath{fill}
      }
      \ifbmb@shadow%
      {
        \color{black!50!bg}
        \pgfsetlinewidth{0pt}
        \pgfpathmoveto{\pgfpoint{\bmb@dimb}{-.5bp}}
        \bmb@dima=\ht\bmb@box%
        \advance\bmb@dima by\bmb@prevheight%
        \advance\bmb@dima by 1bp%
        \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
        \pgfusepath{stroke}
        \bmb@temp=\bmb@dima
        \advance\bmb@dima by 1bp%
        \color{black!31!bg}
        \pgfpathmoveto{\pgfpoint{\bmb@dimb}{\bmb@temp}}
        \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
        \pgfusepath{stroke}
        \advance\bmb@dima by 1bp%
        \advance\bmb@temp by 1bp%
        \color{black!19!bg}
        \pgfpathmoveto{\pgfpoint{\bmb@dimb}{\bmb@temp}}
        \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
        \pgfusepath{stroke}
        \advance\bmb@dima by 1bp%
        \advance\bmb@temp by 1bp%
        \color{black!6!bg}
        \pgfpathmoveto{\pgfpoint{\bmb@dimb}{\bmb@temp}}
        \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
        \pgfusepath{stroke}
        \advance\bmb@dima by 1.5bp%
        \advance\bmb@temp by 1bp%
        \color{bg}
        \pgfpathmoveto{\pgfpoint{\bmb@dimb}{\bmb@temp}}
        \pgfpathlineto{\pgfpoint{\bmb@dimb}{\bmb@dima}}
        \pgfusepath{stroke}
      }
      \fi
    \end{pgfpicture}%
    \box\bmb@box%
  }%
  \ifbmb@shadow%
    \vskip4bp minus 2bp%
  \else%
    \vskip2bp%
  \fi%
  \egroup% of \vbox\bgroup
}

%% ----------------------------------------
%% | /KIT-style semi-rounded beamer boxes |
%% ----------------------------------------

%% use this for setting the total page number
\newcommand{\beginbackup}{
   \newcounter{framenumbervorappendix}
   \setcounter{framenumbervorappendix}{\value{framenumber}}
}
\newcommand{\backupend}{
   \addtocounter{framenumbervorappendix}{-\value{framenumber}}
   \addtocounter{framenumber}{\value{framenumbervorappendix}} 
}