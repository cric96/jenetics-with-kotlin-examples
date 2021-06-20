import io.jenetics.Mutator
import io.jenetics.Optimize
import io.jenetics.SinglePointCrossover
import io.jenetics.engine.*
import io.jenetics.util.DoubleRange
import kotlin.math.cos

/** example taken from https://jenetics.io/manual/manual-6.2.0.pdf at page 121*/
fun main() {
    val A = 50.0
    val R = 5.12
    val N = 20
    fun fitness(x : DoubleArray) : Double {
        var value = A * N
        for (i in 1 until N) {
            value = x[i] * x[i] + A * cos(2.0 * Math.PI * x[i])
        }
        return value
    }
    val engine = Engine.builder(::fitness, Codecs.ofVector(DoubleRange.of(-R, R), N))
        .optimize(Optimize.MINIMUM)
        .alterers(
            Mutator(0.03),
            SinglePointCrossover(0.6)
        ).build()

    val statistics = EvolutionStatistics.ofNumber<Double>()

    val result = engine
        .limit { Limits.bySteadyFitness(7) }
        .stream()
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    println(statistics)
    println(result)
}
