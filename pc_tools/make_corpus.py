#!/usr/bin/env python3
"""
make_corpus.py — build corpus.bin for the ESP32 RAG storyteller.

Input: a text file where each non-empty line (or paragraph) is one passage,
or a directory of .txt files. Passages should be 1-3 sentences, self-contained.

  python3 make_corpus.py facts.txt corpus.bin
  python3 make_corpus.py my_notes_dir/ corpus.bin --split paragraph

Copy corpus.bin to the SD card root. With no input file, writes a built-in
starter corpus of ~60 kid-friendly facts so you can test immediately:

  python3 make_corpus.py --starter corpus.bin
"""
import struct, sys, os, argparse

MAGIC = 0x47415253

STARTER = """\
Elephants are the largest land animals. They can weigh as much as four cars and use their trunks to drink, smell, and hug each other.
Honey never spoils. Archaeologists found honey in ancient Egyptian tombs that was still good to eat after three thousand years.
The Moon is slowly drifting away from Earth, about four centimeters every year, roughly the speed your fingernails grow.
Octopuses have three hearts and blue blood. Two hearts pump blood to the gills and one to the rest of the body.
A bolt of lightning is five times hotter than the surface of the Sun.
Penguins cannot fly, but they are amazing swimmers and can dive deeper than two hundred meters to catch fish.
The Eiffel Tower grows about fifteen centimeters taller in summer because heat makes the iron expand.
Sharks existed before trees. Sharks have been swimming in the oceans for more than four hundred million years.
A group of crows is called a murder. Crows are very clever and can use sticks as tools to get food.
Water covers about seventy percent of the Earth, but less than one percent of it is fresh water people can drink.
Your heart beats about one hundred thousand times every day, pumping blood all around your body.
Butterflies taste with their feet. When they land on a flower they can tell right away if it is good to eat.
The Sahara is the largest hot desert in the world, almost as big as the whole United States.
Sound travels about four times faster in water than in air, which is how whales talk across long distances.
A snail can sleep for up to three years if the weather is too dry.
The Great Wall of China is more than twenty thousand kilometers long and took many hundreds of years to build.
Bees tell each other where flowers are by doing a special waggle dance inside the hive.
Mount Everest is the tallest mountain above the sea, but the tallest mountain from base to top is Mauna Kea in Hawaii.
Polar bears have black skin under their white fur, which helps them soak up the warmth of the sun.
There are more stars in the universe than grains of sand on all the beaches of Earth.
A day on Venus is longer than its year. Venus spins so slowly that it goes around the Sun before it finishes one turn.
Giraffes have the same number of neck bones as people, just seven, but each bone is much bigger.
Bananas are berries, but strawberries are not. Scientists sort fruits by how they grow from the flower.
The blue whale is the largest animal that has ever lived, bigger than any dinosaur, and its heart is as big as a small car.
Ants can lift things fifty times heavier than their own bodies, like a child lifting a car.
Rainbows are made when sunlight passes through raindrops and splits into all its hidden colors.
The ESP32 is a tiny computer chip that costs only a few dollars but can connect to WiFi and run little programs.
Dolphins sleep with one half of their brain at a time so they can keep swimming and breathing.
Hot air rises, which is how hot air balloons float up into the sky.
Trees talk to each other underground through networks of fungus that connect their roots.
A cheetah can run as fast as a car on the highway, but only for a short sprint.
The first computers were so big they filled entire rooms, yet your phone today is millions of times faster.
Camels store fat in their humps, not water. The fat gives them energy to cross the desert.
Owls cannot move their eyes, so they turn their whole heads, almost all the way around, to look behind them.
Volcanoes are openings in the Earth where melted rock called magma pushes up from deep below.
The human body has two hundred and six bones, and more than half of them are in the hands and feet.
Maple syrup comes from the sap of maple trees, collected in early spring when days are warm and nights are cold.
In Canada, winters in places like Fort McMurray can reach minus forty degrees, so cold that boiling water thrown in the air turns to snow.
Northern lights happen when tiny particles from the Sun crash into the air high above the Earth and make the sky glow green and purple.
Beavers build dams from sticks and mud to make ponds where they can stay safe from other animals.
A robot is a machine that can sense the world, make decisions, and move or act on its own.
Magnets have two poles, north and south. Opposite poles pull together and matching poles push apart.
Seeds can sleep in the ground for many years and only wake up and grow when water and warmth arrive.
The Sun is a star, just like the tiny lights in the night sky, only much, much closer to us.
Spiders are not insects. Insects have six legs but spiders have eight, and they make silk to build their webs.
Sea turtles return to the same beach where they were born to lay their own eggs, even after swimming thousands of kilometers.
The Pacific Ocean is the biggest ocean, so large that all the land on Earth could fit inside it.
Echoes happen when sound bounces off walls or mountains and comes back to your ears a moment later.
Fireflies make light inside their bodies with a special chemical, and they flash to talk to each other.
A year is how long the Earth takes to travel once around the Sun, about three hundred sixty five days.
Glaciers are rivers of ice that move very slowly, carving valleys into mountains over thousands of years.
Hummingbirds can fly backwards and their wings beat more than fifty times every second.
Recycling means turning old things like bottles and paper into new things instead of throwing them away.
Earthquakes happen when giant pieces of the Earth's crust slip and slide against each other.
The library of Alexandria was one of the biggest libraries of the ancient world, full of scrolls instead of books.
Tigers have striped skin, not just striped fur. Every tiger's stripes are different, like fingerprints.
Clouds are made of millions of tiny drops of water so light that they float in the sky.
A compass needle points north because the whole Earth acts like a giant magnet.
Frogs drink water through their skin instead of their mouths.
Astronauts grow a little taller in space because without gravity their spines stretch out.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="facts .txt file or directory")
    ap.add_argument("out")
    ap.add_argument("--starter", action="store_true", help="use built-in corpus")
    ap.add_argument("--split", choices=["line", "paragraph"], default="line")
    a = ap.parse_args()

    if a.starter or not a.input:
        text = STARTER
    elif os.path.isdir(a.input):
        text = "\n".join(open(os.path.join(a.input, f), encoding="utf-8").read()
                          for f in sorted(os.listdir(a.input)) if f.endswith(".txt"))
    else:
        text = open(a.input, encoding="utf-8").read()

    if a.split == "paragraph":
        passages = [p.strip().replace("\n", " ") for p in text.split("\n\n")]
    else:
        passages = [l.strip() for l in text.splitlines()]
    passages = [p for p in passages if len(p) > 10]

    blob = bytearray(struct.pack("<II", MAGIC, len(passages)))
    for p in passages:
        b = p.encode("utf-8")[:1000]
        blob += struct.pack("<H", len(b)) + b + b"\0"
    open(a.out, "wb").write(blob)
    print(f"wrote {a.out}: {len(passages)} passages, {len(blob)/1024:.1f} KB")
    print("copy it to the SD card root as corpus.bin")

if __name__ == "__main__":
    main()
