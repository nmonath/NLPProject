#in future would be nice to have a test class to execute tests of the application
import Dependency
dep = Dependency.ReadDependencyParseFile("testFiles/DependencyParsed.txt")
Dependency.Display(dep)