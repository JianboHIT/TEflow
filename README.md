# TEflow (under-developed)
A python3 package for streamlining thermoelectric workflow from materials to devices

## Features
- Model carrier transport
  - Single parabolic band (SPB) model
  - Single Kane band (SKB) model
  - Multiple bands model
  - Customized band model
- Engineering performance of thermoelectric generator[^1]
  - Engineering dimensionless figure of merit (ZT<sub>eng</sub>) and power factor (PF<sub>eng</sub>)
  - Maximum Efficiency (η<sub>max</sub>) and ouput power density (P<sub>d</sub>)
  - R<sub>L</sub>- (external electric load resistance) or I- (electric current density) dependent properties, e.g. output voltage (V), heat flux (Q<sub>hot</sub>)
- Device ZT of thermoelectric generator[^2]
  - Maximum thermoelectric device efficiency
  - Optimized relative current density $u$
  - Thermoelectric potential $\Phi$
- Thermoelectric data manipulation
  - Thermoelectric data interpolation and extrapolation
  - Cut-off thermoelectric data at the threshold temperature
  - Join and rearrange parallel data files
  - Mix parallel data files with linear combination

<br/><br/>
#### References

[^1]: Kim, H. S., Liu, W., Chen, G., Chu, C. W., & Ren, Z. (2015). Relationship between thermoelectric figure of merit and energy conversion efficiency. 
_Proceedings of the National Academy of Sciences_, 112(27), 8205-8210. DOI: [10.1073/pnas.1510231112](https://doi.org/10.1073/pnas.1510231112)

[^2]: Snyder, G. J., & Snyder, A. H. (2017). Figure of merit ZT of a thermoelectric device defined from materials properties. 
_Energy & Environmental Science_, 10(11), 2280-2283. DOI: [10.1039/C7EE02007D](https://doi.org/10.1039/C7EE02007D)
