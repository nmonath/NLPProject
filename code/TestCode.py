#in future would be nice to have a test class to execute tests of the application
import Dependency
import Word2VecExecuter
dep = Dependency.ReadDependencyParseFile("testFiles/DependencyParsed.txt")
Dependency.Display(dep)
Word2VecExecuter.Word2VecTrain("tools/word2vec/word2vec-read-only/text8", "tools/word2vec/word2vec-read-only/vecss.bin")
vector = Word2VecExecuter.Word2VecGetVector("tools/word2vec/word2vec-read-only/vecss.bin", "hello")
print(str(vector))