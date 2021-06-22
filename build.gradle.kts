import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

val jeneticsVersion = "6.2.0"
val deepLearningVersion = "1.0.0-M1"
plugins {
    kotlin("jvm") version "1.5.10"
    application
}

group = "me.cric"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.jenetics", "jenetics", jeneticsVersion)
    implementation("io.jenetics", "jenetics.prog", jeneticsVersion)
    implementation("org.deeplearning4j", "deeplearning4j-core", deepLearningVersion)
    implementation("org.nd4j", "nd4j-native-platform", deepLearningVersion)
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile>() {
    kotlinOptions.jvmTarget = "1.8"
}

application {
    mainClassName = "MainKt"
}