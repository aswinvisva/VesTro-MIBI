import matplotlib.pyplot as plt

from src.utils.utils_functions import mkdir_p


class BaseMIBIPlot:

    def __init__(self,
                 x=None,
                 y=None,
                 data=None,
                 hue=None,
                 style=None,
                 color_map=None,
                 palette=None,
                 x_tick_labels=None,
                 y_tick_labels=None,
                 x_axis_label=None,
                 y_axis_label=None,
                 save=True,
                 show=False,
                 save_dir=None,
                 figsize=(22, 10),
                 fig_name="Test",
                 ax=None
                 ):
        """
        Base MIBI Plot

        :param x: str, Column name for x data
        :param y: str, Column name for y data
        :param data: pd.DataFrame, Pandas Dataframe data
        :param hue: str, Column name for categorical variable
        :param style: str, Column name for style
        :param color_map: matplotlib.ColorMap, Color map
        :param palette: matplotlib.Palette, Palette
        :param x_tick_labels: array_like, X tick labels
        :param y_tick_labels: array_like, Y tick labels
        :param save: bool, Save figure
        :param show: bool, Show figure
        :param save_dir: str, Directory to save figure
        :param figsize: tuple, Figure size
        :param fig_name: str, Figure name
        :param ax: matplotlib.SubplotAxes, Matplotlib Axis
        """

        self.y_axis_label = y_axis_label
        self.x_axis_label = x_axis_label
        self.y_tick_labels = y_tick_labels
        self.x_tick_labels = x_tick_labels
        self.data = data
        self.ax = ax
        self.fig_name = fig_name
        self.figsize = figsize
        self.save_dir = save_dir
        self.show = show
        self.save = save
        self.palette = palette
        self.cmap = color_map
        self.style = style
        self.hue = hue
        self.y = y
        self.x = x

    # def x(self):
    #     for feed_idx in range(self.all_feeds_data.shape[0]):
    #         idx = pd.IndexSlice
    #         feed_name = self.all_feeds_metadata.loc[idx[feed_idx, 0], "Feed Name"]
    #
    #         feed_dir = "%s/%s" % (parent_dir, feed_name)
    #         mkdir_p(feed_dir)
    #
    #         feed_features = self.all_samples_features.loc[self.all_samples_features["Data Type"] == feed_name]

    def make_figure(self, **kwargs):
        """
        Create a plot

        :return:
        """

        raise NotImplementedError("Implement Plot functionality!")

    def savefig_or_show(self, save_dir: str, figure_name: str, ax=None):
        """
        Save Figure or Show Figure

        :return:
        """

        if self.save:
            mkdir_p(self.save_dir)

            if ax is not None:
                ax.savefig(save_dir + '/%s.png' % figure_name, bbox_inches='tight')
            else:
                plt.savefig(save_dir + '/%s.png' % figure_name, bbox_inches='tight')

        if self.show:
            plt.show()

        if self.save:
            plt.clf()
