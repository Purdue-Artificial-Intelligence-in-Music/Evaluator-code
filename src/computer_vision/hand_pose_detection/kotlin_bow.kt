import org.opencv.core.Mat
import kotlin.math.*

class kotlin_bow {

    private var frameNum: Int = 0
    private val stringYCoordHeights: MutableList<Int> = mutableListOf()
    private var bowPoints: MutableList<MutableList<Int>> = mutableListOf(
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0)
    )    
    private var stringPoints: MutableList<MutableList<Int>> = mutableListOf(
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0),
        mutableListOf(0, 0)
    )    
    private var yLocked: Boolean = false
    private var frameCounter: Int = 0
    private val numWaitFrames: Int = 11
    private var yAvg: MutableList<Int> = mutableListOf(0, 0)
    private var bowRepeat: Int = 0
    private var stringRepeat: Int = 0
    // TODO: Add TFLite model once available
    

    private fun update_points(stringBox: MutableList<Int>, bowBox: MutableList<Int>) {
    }

    private fun get_midline(): MutableList<Int> {
    }

    private fun get_vertical_lines(): MutableList<MutableList<Int>> {
    }

    private fun intersects_vertical(linearLine: MutableList<Int>, verticalLines: MutableList<Int>): MutableList<Int> {
    }

    private fun sort_string_points(pts: MutableList<Int>): MutableList<MutableList<Int>> {
    }

    private fun bow_height_intersection(intersectionPoints: MutableList<Int>, verticalLines: MutableList<Int>): Int {
    }

    private fun average_y_coordinates(stringBoxCoords: MutableList<Int>) {
        /*
        Recalculate and updates new string Y coordinate heights for every
        n frames.
         */


        // sort coordinate points so points in consistent order
        val sortedCoords : MutableList<Int> = sort_string_points(stringBoxCoords)
        
        // increase frame counter. create and add y coord list of each frame to one list
        frameCounter += 1
        y_coords :  MutableList<Int> = null
        val yCoords = sortedCoords.map { it[1] } // get Y values from each point
        stringYCoordHeights.add(yCoords)

        // recalculate new y coordinate heights when certain number of frames past
        if (frameCounter % numWaitFrames == 0) {

            // get median of coordinates
            val topLeftAvg: Double = median(stringYCoordHeights.map { it[0] })
            val topRightAvg: Double = median(stringYCoordHeights.map { it[1] })
            val botRightAvg: Double = median(stringYCoordHeights.map { it[2] })
            val botLeftAvg: Double = median(stringYCoordHeights.map { it[3] })

            // map them to string points
            stringPoints[0][1] = topLeftAvg.toInt()
            stringPoints[1][1] = topRightAvg.toInt()
            stringPoints[2][1] = botRightAvg.toInt()
            stringPoints[3][1] = botLeftAvg.toInt()
            yAvg = mutableListOf(topLeftAvg, topRightAvg)
            y_locked = True
            stringYCoordHeights = mutableListOf()
        }
    }

    /*
        Returns median item of a list
     */
    private fun median(values: List<Int>): Double {
        if (values.isEmpty()) throw IllegalArgumentException("Empty list has no median.")

        val sorted = values.sorted()
        val middle = sorted.size / 2

        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle].toDouble()
        }
    }

    /*
    converts radians to degrees
     */
    private fun degrees(radians: Double): Double {
        return radians * (180.0 / PI)

    }
    /* 
    classifies bow angle relative to two vertical lines of string box
     */
    private fun bow_angle(bowLine: MutableList<Int>, verticalLines: MutableList<Int>): Int {
]
        // flexibility of angle relative to 90 degrees
        val max_angle = 15

        // grab bow line and vertical lines
        val m_bow : Double = bowLine[0].toDouble()
        val m1 : Double = verticalLines[0].toDouble()
        val m2 : Double = verticalLines[1].toDouble()  // assuming format: [m1, b1, m2, b2]

        // calculate angles formed for each vertical line's intersection with bow line
        val angle_one : Double = abs(degrees(atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        val angle_two : Double = abs(degrees(atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        val min_angle : Double = min(angle_one, angle_two)

        return if (min_angle > max_angle) 1 else 0  // 1 = Wrong Angle, 0 = Correct Angle
    }

    fun display_classification(result: Int, opencvFrame: Mat) {
    }

    data class BowResults(
        val classification: Int?,
        val bow: MutableList<MutableList<Int>>? ,
        val string: MutableList<MutableList<Int>>?,
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
            if (results.bow == null && bowRepeat < 5 && bowPoints != null) {
                return_bow.classification = -1
                bowRepeat++
                return_bow.bow = bowPoints
        }
        if (results.bow != null) {
            return_bow.bow = results.bow
            if (results.string == null && stringRepeat < 5 && stringPoints != null) {
                return_bow.classification = -1
                stringRepeat++
                return_bow.string = stringPoints
            }
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