import ij._
import ij.io.FileOpener
import ij.io.FileInfo
import ij.io.Opener
import ij.plugin.filter.BackgroundSubtracter
import ij.io.FileSaver
import java.io.File
import scala.util.control.Breaks._

/**
  * 
  *
  * @param source path to source file or directory
  * @param destination path to destination directory
  */
class BatchProcess(val source:String, val destination:String) {
    breakable {
        val s = new File(source)
        val d = new File(destination)
        if(!d.isDirectory()) {  // destination directory exists?
            Console.err.println("Destination directory does not exist!")
            break()             // if not, terminate!
        }
        if(s.exists()) {
            if(s.isFile()) {
                process(s, new File(s"${d.getAbsolutePath()}/${s.getName()}"))
            }
            if(s.isDirectory()) {
                for(f <- s.listFiles()) {
                    process(f, new File(s"${d.getAbsolutePath()}/${f.getName()}"))
                }
            }
        }
    }
    

    /**
      * 
      *
      * @param sourcefile source image file
      * @param destfile destination image file
      */
    def process(sourcefile:File, destfile:File): Unit = {
        println(s"${sourcefile.getAbsoluteFile()} ---> ${destfile.getAbsolutePath()}")
        val opener = new Opener()
        val im = opener.openImage(sourcefile.getAbsolutePath())
        val processor = im.getProcessor()
        val bs = new BackgroundSubtracter()
        bs.rollingBallBackground(processor, 250, false, false, false, false, true)
        processor.resetMinAndMax()
        val saver = new FileSaver(im)
        saver.saveAsTiff(destfile.getAbsolutePath())
    }
    
}
