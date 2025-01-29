const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
    // mode: 'production',
    mode: 'development',
    entry: './src/index.js',
    devtool: 'inline-source-map',
    devServer: {
        static: './dist',
    },
    performance: {
        maxEntrypointSize: 100000000,
        maxAssetSize: 100000000,
    },
    plugins: [
        new HtmlWebpackPlugin({
          title: 'Output Management',
          template: './src/index.html'
        }),
      ],
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
        clean: true,
    },
    optimization: {
        runtimeChunk: 'single',
    },
    module: {
        rules: [
            {
                test: /\.(jpg|png)$/,
                use: {
                    loader: 'file-loader',
                }
            }
        ]
    }
}