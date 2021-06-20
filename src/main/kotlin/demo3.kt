import io.jenetics.Mutator
import io.jenetics.Optimize
import io.jenetics.SinglePointCrossover
import io.jenetics.engine.*
import io.jenetics.util.DoubleRange
import kotlin.math.cos
import kotlin.math.sin

/* example taken by the book https://jenetics.io/manual/manual-6.2.0.pdf at page 120 */
fun main() {
    fun fitness(x : Double) = cos(sin(x) + 0.5) * cos(x)
    val engine = Engine
        .builder(::fitness, Codecs.ofScalar(DoubleRange.of(0.0, 2 * Math.PI)))
        .executor(Runnable::run)
        .optimize(Optimize.MINIMUM)
        .alterers(
            Mutator(0.003),
            SinglePointCrossover(0.6)
        ).build()

    val statistics = EvolutionStatistics.ofNumber<Double>()

    val best = engine
        .limit{ Limits.bySteadyFitness(7) }
        .limit(100)
        .stream()
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    println(statistics)
    println(best)
}
