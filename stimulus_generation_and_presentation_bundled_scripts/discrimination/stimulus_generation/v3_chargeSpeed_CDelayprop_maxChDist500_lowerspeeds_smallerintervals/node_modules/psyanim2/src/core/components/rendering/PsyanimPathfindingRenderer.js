import Phaser from 'phaser';

import PsyanimComponent from "../../PsyanimComponent.js";

export default class PsyanimPathfindingRenderer extends PsyanimComponent {

    lineWidth;
    radius;

    pathfinder;

    pathColor;
    radiusColor;

    constructor(entity) {

        super(entity);

        this.lineWidth = 2;
        this.radius = 12;
    
        this.pathfinder = null;
    
        this.pathColor = 0xBF40BF;
        this.radiusColor = 0xcccccc;

        this._gridWidth = this.scene.game.scale.width;
        this._gridHeight = this.scene.game.scale.height;

        this._cellSize = 20;

        this._grid = this.scene.add.grid(
            this._gridWidth / 2, this._gridHeight / 2,
            this._gridWidth, this._gridHeight,
            this._cellSize, this._cellSize,
            0xffffff, 0.0, 0x000000, 0.3
        );

        this._grid.visible = false;

        this._pathGraphics = this.entity.scene.add.graphics();
        this._radiusGraphics = this.entity.scene.add.graphics();

        this._pathGraphics.depth = -1;
        this._radiusGraphics.depth = -2;
        this._grid.depth = -3;

        this._line = new Phaser.Geom.Line(0, 0, 0, 0);
    }

    setLineDepth(value) {

        this._pathGraphics.depth = value;
    }

    setRadiusDepth(value) {

        this._radiusGraphics.depth = value;
    }

    onDisable() {

        super.onDisable();

        this._pathGraphics.clear();
        this._radiusGraphics.clear();
    }

    setGridVisible(visible) {

        this._grid.visible = visible;
    }

    update(t, dt) {

        super.update(t, dt);

        // update parameters from pathfinder if it has a grid
        if (this.pathfinder.grid && this.pathfinder.grid.cellSize)
        {
            this._grid.cellHeight = this.pathfinder.grid.cellSize;
            this._grid.cellWidth = this.pathfinder.grid.cellSize;    
        }

        let pathSegments = this.pathfinder.currentPath.segments;

        // draw path
        this._pathGraphics.clear();
        this._radiusGraphics.clear();

        this._pathGraphics.lineStyle(this.lineWidth, this.pathColor);
        this._radiusGraphics.lineStyle(this.radius, this.radiusColor);

        for (let i = 0; i < pathSegments.length; ++i)
        {
            let segment = pathSegments[i];

            let fromPoint = segment.p1;
            let toPoint = segment.p2;

            this._line.setTo(
                fromPoint.x, fromPoint.y,
                toPoint.x, toPoint.y
            );

            this._pathGraphics.strokeLineShape(this._line);
            this._radiusGraphics.strokeLineShape(this._line);
        }
    }
}