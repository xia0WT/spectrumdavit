from scipy import signal
import matplotlib.pyplot as plt
class FindPeaks(object):
    def __init__(self,
                 two_theta,
                 intensity,
        ):
        self.two_theta = two_theta
        self.intensity = intensity
        self.fig , self.ax = plt.subplots()
        
    def find_peak(self,
                  width :list[int,int] = None,
                  distance :int = None,
                  height = 5,
                  prominence = 5,
                  resolution = 200,
                  truncate_two_theta = 90,
                  norm = 100,
                  plot = True,
                  save_dir = None,
        ):

        two_theta, intensity = self._truncate_norm(self.two_theta,
                                              self.intensity,
                                              threshold=truncate_two_theta,
                                              norm = norm)
        if not width:
            width = [3, int(len(two_theta) /  (2 *resolution)) ** 2]
            
        if not distance:
            distance = int(len(two_theta) / resolution) * 2
        peaks = self._peaks(two_theta,
                            intensity,
                            distance=distance,
                            width=width,
                            height=height,
                            prominence=prominence)

        self.plot_xrd(two_theta, intensity, peaks)
        if plot:
            self.fig.show()
        else:
            plt.close(self.fig)  #plot = False
        if save_dir:
            self.fig.savefig(f"{save_dir}/result.png", dpi =300)
        return peaks
        
    def _peaks(self,
               x,
               y,
               distance,
               width :list[int,int] ,
               height,
               prominence,
        ):
        
        peaks, _ = signal.find_peaks(y, distance=distance, width = width, height = height, prominence= prominence)
        return x[peaks], y[peaks]

    def _truncate_norm(self,
                  two_theta,
                  intensity,
                  threshold,
                  norm,
        ):
        truncate_two_theta = two_theta[two_theta <= threshold]
        truncate_intensity = intensity[two_theta <= threshold]

        return truncate_two_theta, truncate_intensity * norm / truncate_intensity.max()

    def plot_xrd(self,
                 two_theta,
                 intensity,
                 peaks,
        ):
        
        self.ax.plot(two_theta, intensity, color = "r", label="XRD")
        self.ax.vlines(peaks[0], 0, peaks[1], color = "b", label="characterise peaks")
        self.ax.set_ylim(0)
        self.ax.set_ylabel("intensity")
        self.ax.set_xlabel("2θ")
        self.ax.grid()
        self.ax.legend()
