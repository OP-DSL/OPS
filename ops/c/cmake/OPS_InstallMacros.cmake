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

macro(SetLib LibName SRC Links Opts)
  #message(STATUS "LibName ${LibName}")
  add_library(${LibName} ${SRC})
  foreach(Link IN LISTS Links)
    target_link_libraries(${LibName} PRIVATE ${Link})
  endforeach()
  # Additional flags only for this target
  foreach(Opt IN LISTS Opts)
    target_compile_options(${LibName} PRIVATE ${Opt})
  endforeach()
  installtarget(${LibName} ${ConfigPackageLocation})
endmacro()
