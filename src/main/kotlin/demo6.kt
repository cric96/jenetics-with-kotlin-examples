import io.jenetics.Mutator
import io.jenetics.RouletteWheelSelector
import io.jenetics.SinglePointCrossover
import io.jenetics.TournamentSelector
import io.jenetics.engine.*
import io.jenetics.util.ISeq
import io.jenetics.util.RandomRegistry
import java.util.stream.Stream

/** example taken from https://jenetics.io/manual/manual-6.2.0.pdf page 124, knapsack proble*/
data class Item(val size : Double, val value : Double) {
    companion object {
        fun random(bound : Double = 100.0) : Item {
            val random = RandomRegistry.random()
            return Item(random.nextDouble() * bound, random.nextDouble() * bound)
        }
    }
}
fun fitnessFactory(size : Double) : (ISeq<Item>) -> Double = {
    items ->
        val summed = items.fold(Item(0.0, 0.0)) {
                acc: Item, item: Item -> Item(acc.size + item.size, acc.value + item.value)
        }
        if(summed.size <= size) { summed.value } else { 0.0 }
}

fun main() {
    val bound = 100.0
    val steadyLength = 7
    val nItems = 15
    val probabilityMutation = 0.115
    val probabilityCrossover = 0.16
    val populationSize = 500
    val evolutionLength = 100
    val arenaSize = 5
    val knapSackSize = nItems * bound / 3.0
    val items = Stream.generate { Item.random(bound) }
        .limit(nItems.toLong())
        .collect(ISeq.toISeq())

    val codec = Codecs.ofSubSet(items)

    val engine = Engine
        .builder(fitnessFactory(knapSackSize), codec)
        .populationSize(populationSize)
        .survivorsSelector(TournamentSelector(arenaSize))
        .offspringSelector(RouletteWheelSelector())
        .alterers(
            Mutator(probabilityMutation),
            SinglePointCrossover(probabilityCrossover)
        )
        .build()

    val statistics = EvolutionStatistics.ofNumber<Double>()

    val best = engine.stream()
        .limit(Limits.bySteadyFitness(steadyLength))
        .limit(evolutionLength.toLong())
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype())

    val knapsack = codec.decode(best.genotype())

    println(statistics)
    println(best)
    println("Genotype of best item: ${best.genotype()}")

    val fillSize = knapsack.stream()
        .mapToDouble { it.size }
        .sum()

    println("filled ${bound * fillSize / knapSackSize}")
}