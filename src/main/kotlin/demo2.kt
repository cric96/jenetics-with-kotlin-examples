import io.jenetics.DoubleChromosome
import io.jenetics.DoubleGene
import io.jenetics.Genotype
import io.jenetics.Mutator
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.EvolutionStatistics
import io.jenetics.engine.Limits
import io.jenetics.util.RandomRegistry
import java.util.*
import kotlin.math.pow

/**
 * this example use random registry to produce the same result. Use the previous example as basis
 */
fun main() {
    val seed = 42L
    val mutationProbability = 0.6
    val lowerBound = 0.001
    val bound = 10.0
    fun functionToLearn(input : Int) = ((input) * 10) + 7
    val inputRange = (1..100)
    val trainSet = inputRange.map { functionToLearn(it) }
    //fitness function
    fun fitness(genotype: Genotype<DoubleGene>): Double {
        val chromosome = genotype.chromosome()
        val m = chromosome.get(0)
        val q = chromosome.get(1)
        val result = inputRange.map { (it * m.allele()) + q.allele() }
        return result.zip(trainSet).sumOf { (result, truth) -> (truth - result).pow(2) }
    }
    val statistics = EvolutionStatistics.ofNumber<Double>()
    val factory : Genotype<DoubleGene> = Genotype.of(DoubleChromosome.of(-bound, bound, 2))
    val engine = Engine.builder(::fitness, factory)
        .executor(Runnable::run) //to reproducibility reasons
        .alterers(Mutator(mutationProbability))
        .minimizing()
        .build()

    val result = RandomRegistry.with(Random(seed)) {
        engine.limit { Limits.byFitnessThreshold(lowerBound) }
            .stream()
            .peek (statistics) //utility
            .collect(EvolutionResult.toBestGenotype()) //it can be improved with pimping
    }

    println(statistics)
    result.forEach {
        run {
            val chromosome = it
            val m = chromosome.get(0)
            val q = chromosome.get(1)
            inputRange.forEach { input -> println("$input, ${input * m.allele() + q.allele()}") }
        }
    }
}