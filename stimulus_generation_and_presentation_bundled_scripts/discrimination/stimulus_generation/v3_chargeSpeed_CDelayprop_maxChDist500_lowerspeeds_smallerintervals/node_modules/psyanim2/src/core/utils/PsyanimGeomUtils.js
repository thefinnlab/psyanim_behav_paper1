import PsyanimConstants from "../PsyanimConstants.js";

export default class PsyanimGeomUtils {

    /**
     *  NOTE: when generating a texture from any kind of Phaser graphics object, it's critical to place it at the center of 
     *  your viewport if you want the associated texture to be centered and not clipped.
     */

    static computeTriangleVertices(base, altitude) {

        return [

            {x: -altitude * (1/3), y: -base / 2},
            {x: altitude * (2/3), y: 0 },
            {x: -altitude * (1/3), y: base / 2},
        ];
    }

    /**
     * 
     * @param {*} scene 
     * @param {*} textureKey 
     * @param {*} geomParams
     * @param {*} color 
     */
    static generateTriangleTexture(scene, textureKey, geomParams = { base: 12, altitude: 24}, color = 0x0000ff) {

        let verts = PsyanimGeomUtils.computeTriangleVertices(geomParams.base, geomParams.altitude);

        // some funky stuff happens, so I was able to reverse-engineer how to align this w/ matter-js collision
        // and it involves increasing the canvas size by twice a magic number and shifting the geo by half that
        let magicNumber = 1/6;

        let textureWidth = geomParams.altitude * (1 + 2 * magicNumber);
        let textureHeight = geomParams.base;

        let translatedVerts = [
            { x: verts[0].x + ((1/2) + magicNumber) * geomParams.altitude, y: verts[0].y + (1/2) * geomParams.base },
            { x: verts[1].x + ((1/2) + magicNumber) * geomParams.altitude, y: verts[1].y + (1/2) * geomParams.base },
            { x: verts[2].x + ((1/2) + magicNumber) * geomParams.altitude, y: verts[2].y + (1/2) * geomParams.base }
        ];

        let graphics = scene.add.graphics();
        graphics.fillStyle(color);

        graphics.fillTriangle(
            translatedVerts[0].x, translatedVerts[0].y,
            translatedVerts[1].x, translatedVerts[1].y,
            translatedVerts[2].x, translatedVerts[2].y);

        graphics.generateTexture(textureKey, textureWidth, textureHeight);
        graphics.destroy();
    }

    /**
     * 
     * @param {*} scene 
     * @param {*} textureKey 
     * @param {*} geomParams
     * @param {*} color 
     */
    static generateCircleTexture(scene, textureKey, geomParams = { radius: 12 }, color = 0xffff00) {

        let graphics = scene.add.graphics();
        graphics.fillStyle(color);

        graphics.fillCircle(geomParams.radius, geomParams.radius, geomParams.radius);

        let diameter = geomParams.radius * 2;

        graphics.generateTexture(textureKey, diameter, diameter);
        graphics.destroy();
    }

    static generateRectangleTexture(scene, textureKey, geomParams = { width: 60, height: 30 }, color = 0xffff00) {

        let graphics = scene.add.graphics();
        graphics.fillStyle(color);

        graphics.fillRect(
            0, 0, geomParams.width, geomParams.height
        );

        graphics.generateTexture(textureKey, geomParams.width, geomParams.height);
        graphics.destroy();
    }

    static createTriangleSprite(scene, textureKey, 
        geomParams = { x: 400, y: 300, base: 12, altitude: 24 },
        bodyOptions = { collisionFilter: PsyanimConstants.DEFAULT_SPRITE_COLLISION_FILTER }) {

        let verts = PsyanimGeomUtils.computeTriangleVertices(geomParams.base, geomParams.altitude);

        let sprite = scene.matter.add.sprite(geomParams.x, geomParams.y, textureKey, null);

        sprite.setBody({
            type: 'fromVertices',
            verts: verts
        }, bodyOptions);

        return sprite;
    }

    static createCircleSprite(scene, textureKey, 
        geomParams = { x: 400, y: 300, radius: 12 }, 
        bodyOptions = { collisionFilter: PsyanimConstants.DEFAULT_SPRITE_COLLISION_FILTER }) {

        let sprite = scene.matter.add.sprite(geomParams.x, geomParams.y, textureKey);

        sprite.setBody({ 
            type: 'circle', 
            radius: geomParams.radius 
        }, bodyOptions);

        return sprite;
    }
}