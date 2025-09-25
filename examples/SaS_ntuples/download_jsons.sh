mkdir jsons

# Download 2024 golden JSON and pileup JSON
curl https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json -o jsons/Cert_Collisions2024_378981_386951_Golden.json
cp /afs/cern.ch/user/p/pagaigne/public/pileup_jsons/puWeights_2024.json.gz jsons/
