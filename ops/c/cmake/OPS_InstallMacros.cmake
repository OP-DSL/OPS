macro(InstallTarget Name Pos)
  install(
    TARGETS ${Name}
    EXPORT ${Name}_targets                                        
    LIBRARY DESTINATION lib                                           
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin                                           
    INCLUDES
    DESTINATION include)
  install(
    EXPORT ${Name}_targets                                        
    FILE ${Name}_targets.cmake                                    
    NAMESPACE OPS::
    DESTINATION ${Pos})
endmacro()

macro(SetLib LibName SRC Links)
  message(STATUS "SRC MACRO: ${SRC}")
  add_library(${LibName} ${SRC})
  target_link_libraries(${LibName} PRIVATE ${Links})
  installtarget(${LibName} ${ConfigPackageLocation})
endmacro()
