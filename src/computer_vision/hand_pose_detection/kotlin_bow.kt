import org.opencv.core.Mat

class kotlin_bow {

    private var frameNum: Int = 0
    private val stringYCoordHeights: ArrayList<Int> = arrayListOf()
    private var bowPoints: List<List<Int>>? = null
    private var stringPoints: List<List<Int>>? = null
    private var yLocked: Boolean = false
    private var frameCounter: Int = 0
    private val numWaitFrames: Int = 11
    private var yAvg: ArrayList<Int> = arrayListOf(0, 0)
    private var bowRepeat: Int = 0
    private var stringRepeat: Int = 0
    // TODO: Add TFLite model once available

    private fun update_points(stringBox: ArrayList<Int>, bowBox: ArrayList<Int>) {
    }

    private fun get_midline(): ArrayList<Int> {
    }

    private fun get_vertical_lines(): ArrayList<ArrayList<Int>> {
    }

    private fun intersects_vertical(linearLine: ArrayList<Int>, verticalLines: ArrayList<Int>): ArrayList<Int> {
    }

    private fun sort_string_points(pts: ArrayList<Int>): ArrayList<ArrayList<Int>> {
    }

    private fun bow_height_intersection(intersectionPoints: ArrayList<Int>, verticalLines: ArrayList<Int>): Int {
    }

    private fun average_y_coordinates(stringBox: ArrayList<Int>) {
    }

    private fun bow_angle(bowLine: ArrayList<Int>, verticalLines: ArrayList<Int>): Int {
    }

    fun display_classification(result: Int, opencvFrame: Mat) {
    }

    data class BowResults(
        val classification: Int?,
        val bow: List<List<Int>>? ,
        val string: List<List<Int>>?,
        val angle: Int?
    )

    fun processFrame(opencvFrame: Mat): BowResults {
        reuturn_bow = BowResults(
            classification = null,
            bow = null,
            string = null,
            angle = null
        )
        results = model(opencvFrame) //THIS IS NOT THE CORECT WAY< WILL NEED PREPROCESSING AND POSTPROCESSING + LiteRT
        if (results.isEmpty()) {
            return return_bow
        }
        // ASSUMING RESULTS IS AN OBJECT WITH TWO NESTED ARRAYLISTS OF INTS
        // results.bow = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        // results.string = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        // both could be null
        if (string_coords == null && bow_coords == null) {
            return_bow.classification = -2
            return return_bow
        }
        if (results.string != null) {
            //need to do averaging of top two y coords
            return_bow.string = results.string
            average_y_coordinates(results.string)
        }
        if (results.bow != null) {
            return_bow.bow = results.bow
        }
        if (results.bow != null && results.string != null) {
            update_points(results.string, results.bow)
            val midlines = get_midline()
            val vert_lines = get_vertical_lines()
            val intersect_points = intersects_vertical(midlines, vert_lines)
            return_bow.angle = bow_angle(midlines, vert_lines)
            return_bow.classification = intersect_points
            return return_bow

        } else {
            return_bow.classification = -1
            return return_bow
        }

    }
} 