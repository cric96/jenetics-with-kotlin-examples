import io.jenetics.EnumGene
import io.jenetics.PartiallyMatchedCrossover
import io.jenetics.SwapMutator
import io.jenetics.engine.*
import io.jenetics.util.ISeq
import io.jenetics.util.MSeq
import io.jenetics.util.RandomRegistry
import java.util.function.Function
import java.util.stream.IntStream
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/*
val p1 = p.get(i)
val p2 = p.get((i + 1) % p.size())
hypot(p1[0] - p2[0], p1[1] - p2[1])
 */

/** example taken from https://jenetics.io/manual/manual-6.2.0.pdf page 124, traveling problem */

class TravelingSalesman(val points : ISeq<DoubleArray>) : Problem<ISeq<DoubleArray>, EnumGene<DoubleArray>, Double> {
    override fun fitness(): Function<ISeq<DoubleArray>, Double> {
        return Function<ISeq<DoubleArray>, Double> {
                p : ISeq<DoubleArray> -> IntStream.range(0, p.length()).mapToDouble { i ->
                    val p1 = p.get(i)
                    val p2 = p.get((i + 1) % p.size())
                    hypot(p1[0] - p2[0], p1[1] - p2[1]) }
                .sum()
        }
    }

    override fun codec(): Codec<ISeq<DoubleArray>, EnumGene<DoubleArray>> = Codecs.ofPermutation(points)

    companion object {
        fun of(stops: Int, radius : Double) : TravelingSalesman {
            val points = MSeq.ofLength<DoubleArray>(stops)
            val delta = Math.PI * 2.0 / stops

            for (i in 0 until stops) {
                val alpha = delta * i
                val x = cos(alpha) * radius + radius
                val y = sin(alpha) * radius + radius
                points.set(i, doubleArrayOf(x, y))
            }
            val random = RandomRegistry.random()
            for(j in (stops - 1).downTo(1)) {
                val i = random.nextInt(j + 1)
                val tmp = points.get(i)
                points.set(i, points.get(j))
                points.set(j, tmp)
            }

            return TravelingSalesman(points.toISeq())
        }
    }
}


fun main() {
    val stops = 20
    val R = 10.0
    val maxAge = 11
    val populationSize = 500
    val mutationProbability = 0.2
    val crossoverProbability = 0.35
    val steadyLength = 25
    val lengthOfExploration = 250
    val minPathLength = 2.0 * stops * R * sin(Math.PI / stops)

    val travelingSalesman = TravelingSalesman.of(stops, R)
    val engine = Engine.builder(travelingSalesman)
        .minimizing()
        .maximalPhenotypeAge(maxAge.toLong())
        .populationSize(populationSize)
        .alterers(
            SwapMutator(mutationProbability),
            PartiallyMatchedCrossover(crossoverProbability)
        )
        .build()

    val statistics = EvolutionStatistics.ofNumber<Double>()

    val best = engine.stream()
        .limit(Limits.bySteadyFitness(steadyLength))
        .limit(lengthOfExploration.toLong())
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    println(statistics)
    println("Best min path : $minPathLength")
    println("Founded min path : ${best.fitness()}")
}