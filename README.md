# snowglobe

Fork of [Astrabit-ST/snowglobe](https://github.com/Astrabit-ST/snowglobe) with updated SDL library. SDL at commit `7df1cab` in the original repository doesn't compile successfully, and also native version supplied with OneShot: Frostide doesn't work under Wayland.

To make it look better under Hyprland add those lines to your config:
```
windowrulev2 = float, class:snowglobe
windowrulev2 = noblur, class:snowglobe
windowrulev2 = noshadow, class:snowglobe
windowrulev2 = noborder, class:snowglobe
```
