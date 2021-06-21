import io.jenetics.DoubleChromosome
import io.jenetics.DoubleGene
import io.jenetics.Genotype
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.EvolutionStatistics
import io.jenetics.engine.Limits
import kotlin.math.pow

/**
 * evolutionary algorithm with neural networks.
 * It contains a simple implementation of a neural network
 * EV learn the and function
 */
interface Layer {
    val input : Int get() = weights[0].size
    val output : Int get() = weights.size
    val next : Layer?
    val weights : Array<Array<Double>>
    val activation : (Double) -> Double
    operator fun invoke(inputs : Array<Double>) : Array<Double> {
        require(inputs.size == input)
        return weights.map {
            val combination = inputs.zip(it).sumOf { (w, i) -> w * i }
            activation(combination)
        }.toTypedArray()
    }
}

class DenseLayer(override val weights : Array<Array<Double>>, override val activation : (Double) -> Double, override val next : Layer) : Layer
class Output(override val weights : Array<Array<Double>>, override val activation : (Double) -> Double) : Layer {
    override val next: Layer?
        get() = null
}

class Network(val input : Layer) {
    private fun invokeAndProgress(input : Array<Double>, layer : Layer) : Array<Double> {
        val result = layer(input)
        val next = layer.next
        return if(next != null) {
            invokeAndProgress(result, next)
        } else {
            result
        }
    }
    operator fun invoke(inputs : Array<Double>) : Array<Double> {
        return invokeAndProgress(inputs, input)
    }
}
fun identity(input : Double) = input
fun relu(input : Double) = if(input > 0) { input } else { 0.0 }
fun main() {
    val inputSize = 2
    val range = 2.0
    val hiddenNeuron = 2
    val output = 1
    val steadySize = 10
    val populationSize = 500
    val dataSet = mapOf(
        Pair(Pair(1.0, 0.0), 0),
        Pair(Pair(0.0, 1.0), 0),
        Pair(Pair(1.0, 1.0), 1),
        Pair(Pair(0.0, 0.0), 0),
    )

    fun networkFrom(genotype: Genotype<DoubleGene>) : Network {
        val hiddenChromosomes = genotype.toList().take(hiddenNeuron)
        val outputChromosomes = genotype.toList().drop(hiddenNeuron)
        val hiddenLayerWeight = hiddenChromosomes.map { chromosome -> chromosome.map { it.allele() } }
            .map { it.toTypedArray() }
            .toTypedArray()
        val outputLayerWeight = outputChromosomes.map { chromosome -> chromosome.map { it.allele() } }
            .map { it.toTypedArray() }
            .toTypedArray()
        val output = Output(outputLayerWeight, ::identity)
        val hidden = DenseLayer(hiddenLayerWeight, ::relu, output)
        return Network(hidden)
    }
    fun fitness(genotype : Genotype<DoubleGene>) : Double {
        val network = networkFrom(genotype)
        return dataSet.map { (key, value) -> network(key.toList().toTypedArray())[0] - value }
            .sumOf { it.pow(2) }
    }
    //factory
    val hiddens = (1..hiddenNeuron).map { DoubleChromosome.of(-range, range, inputSize) }
    val outputs = (1..output).map { DoubleChromosome.of(-range, range, hiddenNeuron) }
    val factory = Genotype.of(hiddens + outputs)

    val engine = Engine.builder(::fitness, factory)
        .executor(Runnable::run)
        .populationSize(populationSize)
        .minimizing()
        .build()

    val statistics = EvolutionStatistics.ofNumber<Double>()
    val result = engine
        .limit { Limits.bySteadyFitness(steadySize) }
        .stream()
        .peek(statistics)
        .collect(EvolutionResult.toBestPhenotype()) //it can be improved with pimping

    println(statistics) //unsafe..
    val network = networkFrom(result.genotype())
    dataSet.forEach() { (key, value) -> println(network(key.toList().toTypedArray())[0]) }
}
