import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import HtmlWebpackPlugin from 'html-webpack-plugin';

export default {
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