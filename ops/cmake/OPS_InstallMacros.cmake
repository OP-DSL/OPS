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

macro(SetLib LibName SRC Links Opts Defs Deps)
  message(STATUS "Library to be set ${LibName}")
  add_library(${LibName} ${SRC})
  # Additional flags only for this target
  #foreach(Dep IN LISTS Deps)
  #  message(STATUS "DEP ${Dep}")
  #  add_dependencies(${LibName} ${Dep})
  #endforeach()
  foreach(Link IN LISTS Links)
	  target_link_libraries(${LibName} PRIVATE ${Link})
  endforeach()
  # Additional flags only for this target
  foreach(Opt IN LISTS Opts)
	  target_compile_options(${LibName} PRIVATE ${Opt})
  endforeach()
  # Additional flags only for this target
  foreach(Def IN LISTS Defs)
    message(STATUS "DEF ${Def}")
    target_compile_definitions(${LibName} PRIVATE ${Def})
  endforeach()
  installtarget(${LibName} ${ConfigPackageLocation})
endmacro()
