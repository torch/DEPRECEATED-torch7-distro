# Inspired from /usr/share/autoconf/autoconf/c.m4

INCLUDE(CheckCSourceCompiles)

MESSAGE(STATUS "Checking for inline keyword")
FOREACH(KEYWORD "inline" "__inline__" "__inline")
   IF(NOT DEFINED C_INLINE)
     CHECK_C_SOURCE_COMPILES("typedef int foo_t;
                              static inline foo_t static_foo(){return 0;}
                               foo_t foo(){return 0;}
                               int main(int argc, char *argv[]){return 0;}" C_HAS_${KEYWORD})

     IF(C_HAS_${KEYWORD})
       SET(C_INLINE ${KEYWORD})
     ENDIF(C_HAS_${KEYWORD})
   ENDIF(NOT DEFINED C_INLINE)
ENDFOREACH(KEYWORD)
