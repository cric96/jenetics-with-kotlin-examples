import io.jenetics.DoubleChromosome
import io.jenetics.DoubleGene
import io.jenetics.Genotype
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.Limits
import kotlin.math.pow

/**
 * this example try to solve the linear regression with evolutionary algorithm
 */
fun main() {
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

    val factory : Genotype<DoubleGene> = Genotype.of(DoubleChromosome.of(-bound, bound, 2))
    val engine = Engine.builder(::fitness, factory)
        .minimizing()
        .build()

    val result = engine
        .limit { Limits.byFitnessThreshold(0.01) }
        .stream()
        .peek { println(it.bestFitness()) } //utility
        .collect(EvolutionResult.toBestGenotype()) //it can be improved with pimping

    result.forEach {
        run {
            val chromosome = it
            val m = chromosome.get(0)
            val q = chromosome.get(1)
            inputRange.forEach { input -> println("$input, ${input * m.allele() + q.allele()}") }
        }
    }
}