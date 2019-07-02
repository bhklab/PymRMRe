dnl The Autoconf script for Rlobico package
dnl
dnl A working installation of CPLEX is required
dnl
dnl USAGE: autoconf
dnl
dnl Global variables:
dnl        PKG_CXXFLAGS, PKG_CPPFLAGS and PKG_LIBS
dnl
dnl The resulting configure script implements the following:
dnl
dnl 1. If PKG_CXXFLAGS, PKG_CPPFLAGS and PKG_LIBS are defined, use them and exit
dnl
dnl 2. If user specifies location for the CPLEX include file and library as well as Cpp flags
dnl    we set the global variables accordingly. This can be specified through the following
dnl    arguments (all three MUST be given)
dnl         --with-cplex-include=<include-DIR>
dnl         --with-cplex-cxxflags=<required Cxx flags>
dnl         --with-cplex-lib=<library-DIR> <lib flags>
dnl
dnl 3. If user specifies the location of the CPLEX installation, we will guess at the location of the example file
dnl    i.e. guess the system and libformat and set global variables using the found Makefile. This can be set with
dnl
dnl         --with-cplex-dir=<cplex-DIR>
dnl
dnl 4. If none of the above are given, the script tries to find the location of the CPLEX installation and then does the
dnl    same as 3.

AC_INIT([Rlobico], 0.1.0)

FOUND_ALL = "no"

dnl Check if global variables are already set, if so we are done
if test [-n "$PKG_CPPFLAGS" -a -n "$PKG_LIBS"]; then
   FOUND_ALL = "yes"
else
   if test [-n "$PKG_CPPFLAGS" -o -n "$PKG_LIBS"]; then
      AC_MSG_ERROR([All three of PKG_CXXFLAGS, PKG_CPPFLAGS and PKG_LIBS must be given together])
   fi
fi

dnl get arguments
AC_ARG_WITH([cplex-include],
	      [AS_HELP_STRING([--with-cplex-include],[specify location of cplex include directory])],
	      CPLEX_INCLUDE=$withval,
	      CPLEX_INCLUDE="")

AC_ARG_WITH([cplex-cxxflags],
	      [AS_HELP_STRING([--with-cplex-cxxflags],[specify C flags for compilation of interface])],
	      CPLEX_CXXFLAGS=$withval,
	      CPLEX_CXXFLAGS="")

AC_ARG_WITH([cplex-lib],
	      [AS_HELP_STRING([--with-cplex-lib],[specify location of CPLEX callable library and other libraries])],
	      CPLEX_LIB=$withval,
	      CPLEX_LIB="")

AC_ARG_WITH([cplex-dir],
	      [AS_HELP_STRING([--with-cplex-dir],[specify location of CPLEX installation directory])],
	      CPLEX_DIR=$withval,
	      CPLEX_DIR="")

dnl Check if include flags and libs given using --with arguments
if test ["$FOUND_ALL" = "no"]; then
   if test [-n "$CPLEX_INCLUDE" -a -n "$CPLEX_LIB"]; then
      if test [! -d "$CPLEX_INCLUDE"]; then
         AC_MSG_ERROR([CPLEX include directory $CPLEX_INCLUDE does not exist])
      fi

      PKG_CPPFLAGS = "-I$CPLEX_INCLUDE"
      if test [-n "$CPLEX_CXXFLAGS"]; then
         PKG_CXXFLAGS = "$CPLEX_CXXFLAGS"
      fi

      PKG_LIBS = "$CPLEX_LIB"
      FOUND_ALL = "yes"
    else
      if test [-n "$CPLEX_INCLUDE" -o -n "$CPLEX_LIBS"]; then
         AC_MSG_ERROR([All options --with-cplex-include, --with-cplex-cxxflags and --with-cplex-lib must be given together])
      fi
   fi
fi

dnl No information given try to guess it from location of interactive optimizer
if test ["$FOUND_ALL" = "no" -a -z "$CPLEX_DIR"]; then
   AC_PATH_PROG(CPLEX_BIN, cplex)
   CPLEX_DIR = `AS_DIRNAME(["$CPLEX_BIN"])`
   SYSTEM = `AS_BASENAME(["$CPLEX_DIR"])`
   CPLEX_DIR=`AS_DIRNAME(["$CPLEX_DIR"])`
   CPLEX_DIR=`AS_DIRNAME(["$CPLEX_DIR"])`
   if test [-n "$CPLEX_DIR"]; then
      echo "Trying to use $CPLEX_DIR as CPLEX installation location"
   fi
fi

dnl Use the given installation directory --with-cplex-dir
if test ["$FOUND_ALL" = "no"]; then
   if test [! -d "$CPLEX_DIR/include"]; then
      AC_MSG_ERROR([CPLEX include directory ${CPLEX_DIR}/include does not exist])
   fi

   if test [-z "$SYSTEM"]; then
      ## Set system as the first round
      SYSTEM = `ls -l ${CPLEX_DIR}/lib/${SYSTEM} | awk '($1 ~ /d/) {print $NF}' | head -n1 -`
      if test [! -d "${CPLEX_DIR}/bin/${SYSTEM}"]; then
         AC_MSG_ERROR([CPLEX library location could not be found])
      fi
   fi

   ## Guess libformat: just grab the first that comes up
   LIBFORMAT = `ls -l ${CPLEX_DIR}/lib/${SYSTEM} | awk '($1 ~ /d/) {print $NF}' | head -n1 -`
   if test [-z "$LIBFORMAT" -a ! -d "${CPLEX_DIR}/lib/${SYSTEM}/${LIBFORMAT}"]; then
      AC_MSG_ERROR([CPLEX example Makefile could not be found])
   fi

dnl Check for awk
   AC_PROG_AWK

   CPLEXLIBDIR="${CPLEX_DIR}/lib/${SYSTEM}/${LIBFORMAT}"
   CPLEX_LIBS="-L${CPLEXLIBDIR} `${AWK} 'BEGIN {FS = " = "} /^CLNFLAGS/ {print $2}' ${CPLEX_MAKEFILE}`"
   if test [-z CPLEX_LIBS] ; then
      AC_MSG_ERROR([Could not get library info from CPLEX example Makefile])
   fi

   CPLEX_COPT=`${AWK} 'BEGIN {FS = " = "} /^COPT/ {print $2}' ${CPLEX_MAKEFILE}`
   if test [-z CPLEX_COPT] ; then
      AC_MSG_ERROR([Could not get compiler options from CPLEX example Makefile])
   fi

   PKG_CXXFLAGS="$CPLEX_COPT"
   PKG_CPPFLAGS="-I$CPLEX_DIR/include"
   PKG_LIBS="$CPLEX_LIBS"
   FOUND_ALL="yes"
fi


if test "$FOUND_ALL" = "no" ; then
   AC_MSG_ERROR([No cplex installation found])
fi

dnl Find compiler and compiler flags
: ${R_HOME=`R RHOME`}

if test -z "${R_HOME}" ; then
   echo "could not determine R_HOME"
   exit 1
fi
CC=`"${R_HOME}/bin/R" CMD config CC`
CPP=`"${R_HOME}/bin/R" CMD config CPP`
CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
AC_PROG_CC
AC_PROG_CPP

dnl Check for header and library
LIBS="${PKG_LIBS}"
CXXFLAGS="${CXXFLAGS} ${PKG_CXXFLAGS}"
CPPFLAGS="${CPPFLAGS} ${PKG_CPPFLAGS}"

AC_CHECK_HEADER(ilcplex/cplex.h)
if test "${ac_cv_header_ilcplex_cplex_h}" == no; then
   AC_MSG_ERROR([CPLEX header cplex.h not found]);
fi

AC_SEARCH_LIBS(CPXversion,cplex, ,AC_MSG_ERROR([Could not link CPLEX library using $PKG_LIBS]))

dnl substitute in src/Makevars
AC_SUBST(PKG_CXXFLAGS)
AC_SUBST(PKG_LIBS)
AC_SUBST(PKG_CPPFLAGS)
AC_OUTPUT(src/Makevars)

exit 0
