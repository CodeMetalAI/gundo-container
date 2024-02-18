# gundo-container
for un segundo they were on top of el mundo


the settings file must be copied to where UE5 is looking for the swarm, in windows this location default is:
C:\Users\<username>\Documents\AirSim

when you start a scene in UE5, blue text on the screen should tell you where the file is

In the future, we could update this file https://github.com/CodexLabsLLC/Colosseum/blob/2383b0adfc674d34e68cb05ef163ca1c00479352/docker/run_airsim_image_binary.sh#L65C64-L65C72 in our fork of airsim to point to a custom settings.json: https://github.com/CodexLabsLLC/Colosseum/blob/2383b0adfc674d34e68cb05ef163ca1c00479352/docs/pages/colosseum/docker_ubuntu.md?plain=1#L41. Then rebuild airsim.
