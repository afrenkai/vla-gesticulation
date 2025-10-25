#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 1                      # total number of tasks
#SBATCH --mem=64g                  # allocate 1 GB of memory
#SBATCH -J "dataset scraping"      # name of the job
#SBATCH -o zds_%j.out # name of the output file
#SBATCH -e zds_%j.err # name of the error file
#SBATCH -p short                  # partition to submit to
#SBATCH -t 01:00:00               # time limit of 1 hour

uv run zenodo_scraper.py
