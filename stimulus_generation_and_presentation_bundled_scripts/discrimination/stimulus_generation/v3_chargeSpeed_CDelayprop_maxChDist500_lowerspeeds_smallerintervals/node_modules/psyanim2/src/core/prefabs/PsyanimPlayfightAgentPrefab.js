import PsyanimEntityPrefab from '../PsyanimEntityPrefab.js';

import PsyanimVehicle from '../components/steering/PsyanimVehicle.js';
import PsyanimArriveBehavior from '../components/steering/PsyanimArriveBehavior.js';
import PsyanimSeekBehavior from '../components/steering/PsyanimSeekBehavior.js';
import PsyanimWanderBehavior from '../components/steering/PsyanimWanderBehavior.js';
import PsyanimPlayfightBehavior from '../components/steering/PsyanimPlayfightBehavior.js';
import PsyanimPlayfightAgent from '../components/steering/agents/PsyanimPlayfightAgent.js';

import PsyanimFleeBehavior from '../components/steering/PsyanimFleeBehavior.js';

/**
 *  Prefab for creating `Playfight Agents`.
 * 
 *  A `Playfight Agent` is an agent that simulates the playfighting behavior of two animals.
 * 
 *  It wanders about the world randomly, and at specified intervals, attacks another agent by charging 
 *  at it until they make contact, at which point it returns to wandering until the next time to attack.
 * 
 *  A `Playfight Agent` has the following components:
 * 
 *  `PsyanimVehicle`, `PsyanimArriveBehavior`, `PsyanimAdvancedFleeBehavior`, `PsyanimSeekBehavior`, `PsyanimWanderBehavior`,
 *  `PsyanimPlayfightBehavior`, `PsyanimPlayfightAgent`
 */
export default class PsyanimPlayfightAgentPrefab extends PsyanimEntityPrefab {

    /** Playfight Algorithm Params */

    /**
     *  The break duration determines how long, in milliseconds, the agent wanders before attacking its target again.
     *  @type {Number}
     */
    breakDuration;

    /** Charge Params */

    /**
     *  Maximum speed at which this agent can charge at it's target.
     *  @type {Number}
     */
    maxChargeSpeed;

    /**
     *  Maximum acceleration this agent can reach when charging at it's target.
     *  @type {Number}
     */
    maxChargeAcceleration;

    /**
     *  Distance, in px, from target which the agent will come to rest.
     *  @type {Number}
     */
    innerDecelerationRadius;

    /**
     *  Distance, in px, from target which the agent will begin slowing down.
     *  @type {Number}
     */
    outerDecelerationRadius;

    /** Wander Params */

    /**
     *  Maximum speed at which the agent will wander.
     *  @type {Number}
     */    
    maxWanderSpeed;

    /**
     *  Maximum acceleration the agent can attain during wander.
     *  @type {Number}
     */
    maxWanderAcceleration;

    /**
     *  Radius of the wander circle.
     *  @type {Number}
     */    
    wanderRadius;

    /**
     *  Distance the wander circle is offset from the agent's position.
     *  @type {Number}
     */    
    wanderOffset;

    /**
     *  Maximum number of degrees the wander target can move around the wander circle per frame.
     *  @type {Number}
     */
    maxWanderAngleChangePerFrame;
    
    /** Flee Params */

    /**
     *  Maximum speed this agent can flee from the target.
     *  @type {Number}
     */    
    maxFleeSpeed;

    /**
     *  Maximum acceleration this agent can reach during flee from target.
     *  @type {Number}
     */    
    maxFleeAcceleration;

    /**
     *  Distance from target at which the agent will begin fleeing.
     *  @type {Number}
     */
    panicDistance;

    /**
     *  Enables debug output for this agent's Playfight Behavior.
     *  @type {Number}
     */
    debug;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        this.breakDuration = 2000;

        this.maxChargeSpeed = 9;
        this.maxChargeAcceleration = 0.4;

        this.innerDecelerationRadius = 12;
        this.outerDecelerationRadius = 30;

        this.maxWanderSpeed = 4;
        this.maxWanderAcceleration = 0.2;
        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 20;

        // flee params
        this.maxFleeSpeed = 4;
        this.maxFleeAcceleration = 0.2;
        this.panicDistance = 100;

        this.debug = false;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);
    
        let arrive = entity.addComponent(PsyanimArriveBehavior);
        arrive.maxSpeed = this.maxChargeSpeed;
        arrive.maxAcceleration = this.maxChargeAcceleration;
        arrive.innerDecelerationRadius = this.innerDecelerationRadius;
        arrive.outerDecelerationRadius = this.outerDecelerationRadius;

        let seek = entity.addComponent(PsyanimSeekBehavior);

        let flee = entity.addComponent(PsyanimFleeBehavior);
        flee.maxSpeed = this.maxFleeSpeed;
        flee.maxAcceleration = this.maxFleeAcceleration;
        flee.panicDistance = this.panicDistance;

        let wander = entity.addComponent(PsyanimWanderBehavior);
        wander.seekBehavior = seek;
        wander.maxSeekSpeed = this.maxWanderSpeed;
        wander.maxSeekAcceleration = this.maxWanderAcceleration;
        wander.radius = this.wanderRadius;
        wander.offset = this.wanderOffset;
        wander.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        let playfight = entity.addComponent(PsyanimPlayfightBehavior);
        playfight.breakDuration = this.breakDuration;
        playfight.fleeBehavior = flee;
        playfight.arriveBehavior = arrive;
        playfight.wanderBehavior = wander;
        playfight.debug = this.debug;
        
        let playfightAgent = entity.addComponent(PsyanimPlayfightAgent);
        playfightAgent.playfightBehavior = playfight;
        playfightAgent.vehicle = vehicle;
    }
}
